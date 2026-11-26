"""Vectorised top-down ice hockey for multi-agent reinforcement learning.

Every game is 5 skaters + 1 goalie a side on a 200 x 85 ft rink.  A team's
"coach" may pull the goalie at any moment for an extra attacker (slot 5); the
goalie can only come back at a stoppage.  Rules that are hard-coded:

  * goals (puck fully over the goal line between the posts)
  * offside (immediate whistle: an attacker is past the blue line when his team
    brings the puck into the zone; faceoff outside the blue line)
  * icing (no-touch: released from behind centre, crosses the far goal line
    untouched; faceoff in the offending team's end)
  * goalie freezes in or near his crease -> faceoff in that end
  * faceoffs: puck frozen on the dot for a set-up period while every player must
    be on his own side of the dot; after a goal every skater is re-spawned at a
    random spot in his own half, so no line-up is ever imposed
  * goalies stay within 12 ft of the middle of their goal line, in front of
    it (a stand-in for the crease / trapezoid rules; free-roaming goalies
    learned to wander off and leave the net open)

Nothing about positions, formations or roles is coded anywhere.

All N games run in lock-step; arrays carry a leading game axis.  Entity k is
team k // 7, slot k % 7: slots 0-4 regular skaters, 5 the extra attacker
(on ice only while the goalie is pulled), 6 the goalie.  Each team sees the
game in its own frame, rotated 180 degrees for team 1, so both attack +x and
"left" (+y) means the same thing to both.
"""
import numpy as np

# ---------------------------------------------------------------- geometry (ft)
RINK_X, RINK_Y, CORNER_R = 100.0, 42.5, 28.0
GOAL_X, BLUE_X = 89.0, 25.0
POST_Y, NET_DEPTH = 3.0, 3.33
CREASE_R = 6.0
NEUTRAL_DOT = (20.0, 22.0)
END_DOT = (69.0, 22.0)

# ---------------------------------------------------------------- entities
N_SLOTS = 6            # skater slots per team (5 + extra attacker)
EXTRA = 5
GOALIE = 6
E = 7                  # entities per team
K = 2 * E
TEAM_OF = np.repeat([0, 1], E)
SLOT_OF = np.tile(np.arange(E), 2)
SIGN = np.array([1.0, -1.0])
SIGN_K = np.repeat(SIGN, E)
IS_G = SLOT_OF == GOALIE
# view order: for team T, own entities then opponents
VIEW = np.array([list(range(0, 14)), list(range(7, 14)) + list(range(0, 7))])

# ---------------------------------------------------------------- physics
DT = 0.1                    # one decision
SUBSTEPS = 4
SUB_DT = DT / SUBSTEPS
VMAX_K = np.where(IS_G, 14.0, 30.0)
ACC_K = np.where(IS_G, 40.0, 22.0)
RAD_K = np.where(IS_G, 2.2, 1.5)        # body (goalie incl. pads)
REACH_K = np.where(IS_G, 2.2, 2.5)      # stick / blocking reach
BRAKE = 35.0
DRAG = 0.2
CARRY_SPEED = 0.9
PLAYER_REST = 0.3
HIT_SPEED = 12.0            # closing speed that can knock the puck loose
PUCK_FRICTION = 4.0
PUCK_REST = 0.7
PUCK_OFFSET = 1.5
STEAL_R = 3.0
STEAL_P = 0.025             # per close opponent per sub-step
SHOT_SPEED, PASS_SPEED, DUMP_SPEED = 100.0, 50.0, 75.0
SHOT_SIGMA, PASS_SIGMA, DUMP_SIGMA = 0.05, 0.06, 0.10
SAVE_P = 0.90               # goalie stops a hard puck he gets a piece of
FREEZE_P = 0.5
FREEZE_DIST = 12.0
COOLDOWN = 0.3
SETUP_GOAL, SETUP_STOP = 20, 10   # faceoff set-up (decisions)
GOALIE_RANGE = 12.0         # goalie centre stays within this of the goal mouth
GOALIE_MIN_X = -88.0        # team frame: and in front of the goal line

# ---------------------------------------------------------------- actions
# move: 0 glide, 1 brake, 2..9 accelerate in one of 8 directions (team frame)
N_MOVE = 10
_ang = np.arange(8) * np.pi / 4
DIRS = np.zeros((N_MOVE, 2))
DIRS[2:, 0], DIRS[2:, 1] = np.cos(_ang), np.sin(_ang)
# puck (only when carrying): 0 keep, 1 shoot, 2 dump left, 3 dump right,
# 4+j pass to skater slot j
N_PUCK = 4 + N_SLOTS
EV_NONE, EV_GOAL, EV_OFFSIDE, EV_ICING, EV_FREEZE = 0, 1, 2, 3, 4
ROLE_NAMES = ["LD", "RD", "LW", "C", "RW"]

ENT_F, PUCK_F, GAME_F, AGENT_F = 6, 7, 12, 16
OBS_DIM = 2 * E * ENT_F + PUCK_F + GAME_F + AGENT_F + 2 * K
COACH_DIM = 16


def _norm(v):
    return np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1])


def _dot(a, b):
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def _rot(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def clamp_rink(p, v, r, rest):
    """Keep points of radius r inside the rounded rink; bounce velocity."""
    lx, ly = RINK_X - r, RINK_Y - r
    over = np.abs(p[..., 0]) > lx
    sx = np.sign(p[..., 0])
    p[..., 0] = np.where(over, sx * lx, p[..., 0])
    v[..., 0] = np.where(over & (v[..., 0] * sx > 0), -rest * v[..., 0], v[..., 0])
    over = np.abs(p[..., 1]) > ly
    sy = np.sign(p[..., 1])
    p[..., 1] = np.where(over, sy * ly, p[..., 1])
    v[..., 1] = np.where(over & (v[..., 1] * sy > 0), -rest * v[..., 1], v[..., 1])
    cx, cy = RINK_X - CORNER_R, RINK_Y - CORNER_R
    sx, sy = np.sign(p[..., 0]), np.sign(p[..., 1])
    d = p - np.stack([sx * cx, sy * cy], -1)
    dist = _norm(d) + 1e-9
    lim = np.asarray(CORNER_R - r, dtype=float)
    out = (np.abs(p[..., 0]) > cx) & (np.abs(p[..., 1]) > cy) & (dist > lim)
    if out.any():
        n = d / dist[..., None]
        p[out] = (p - d + n * lim[..., None])[out]
        vn = _dot(v, n)
        hit = out & (vn > 0)
        v[hit] = (v - (1 + rest) * vn[..., None] * n)[hit]
    return p, v


def _net_box(sgn):
    if sgn > 0:
        return GOAL_X, GOAL_X + NET_DEPTH
    return -(GOAL_X + NET_DEPTH), -GOAL_X


def role_assign(xt, yt):
    """Label 5 skaters per frame by rank: two deepest = D (larger y = left),
    the rest F by y (LW, C, RW).  xt, yt: [..., 5] team-frame coordinates.
    Returns role index per slot [..., 5]."""
    order = np.argsort(xt, -1, kind="stable")
    d, f = order[..., :2], order[..., 2:]
    dy = np.take_along_axis(yt, d, -1)
    first_left = dy[..., 0] >= dy[..., 1]
    ld = np.where(first_left, d[..., 0], d[..., 1])
    rd = np.where(first_left, d[..., 1], d[..., 0])
    fy = np.take_along_axis(yt, f, -1)
    fo = np.argsort(-fy, -1, kind="stable")
    fs = np.take_along_axis(f, fo, -1)
    roles = np.empty_like(order)
    for r, idx in enumerate([ld, rd, fs[..., 0], fs[..., 1], fs[..., 2]]):
        np.put_along_axis(roles, idx[..., None], r, -1)
    return roles


class HockeyVecEnv:
    def __init__(self, n_envs=64, game_seconds=180, seed=0, offside=True, icing=True):
        self.n = n_envs
        self.T_steps = int(round(game_seconds / DT))
        self.rng = np.random.default_rng(seed)
        self.offside_on, self.icing_on = offside, icing
        n = n_envs
        self.pos = np.zeros((n, K, 2))
        self.vel = np.zeros((n, K, 2))
        self.facing = np.tile(SIGN_K[None, :, None] * np.array([1.0, 0.0]), (n, 1, 1))
        self.active = np.zeros((n, K), bool)
        self.cool = np.zeros((n, K))
        self.hit = np.zeros((n, K), bool)
        self.ppos = np.zeros((n, 2))
        self.pvel = np.zeros((n, 2))
        self.owner = np.full(n, -1)
        self.last_team = np.full(n, -1)
        self.icing_armed = np.full(n, -1)
        self.shot_live = np.zeros(n, bool)
        self.pass_team = np.full(n, -1)
        self.score = np.zeros((n, 2), int)
        self.pulled = np.zeros((n, 2), bool)
        self.setup = np.zeros(n, int)
        self.fo_pt = np.zeros((n, 2))
        self.bench_sign = np.ones(n)
        self.t = 0
        self.stats = {}
        self.pull_log = []      # (time_left_frac, score_diff) at each pull

    # ------------------------------------------------------------ resets
    def reset(self):
        n = self.n
        self.t = 0
        self.score[:] = 0
        self.pulled[:] = False
        self.bench_sign = self.rng.choice([-1.0, 1.0], n)
        self.active[:] = True
        self.active[:, EXTRA] = False
        self.active[:, E + EXTRA] = False
        self._kickoff(np.ones(n, bool))
        self.stats = {k: np.zeros((n, 2)) for k in
                      ["goals", "shots", "sog", "saves", "passes", "pass_ok", "takeaways",
                       "offside", "icing", "freeze", "pulls", "en_goals_against", "goals_6v5",
                       "poss_seconds"]}
        self.pull_log = []
        return self.obs()

    def _park(self, k_mask):
        self.pos[k_mask] = np.array([0.0, 500.0])
        self.vel[k_mask] = 0.0

    def _kickoff(self, m):
        """Random own-half positions, centre faceoff, for games in mask m."""
        idx = np.where(m)[0]
        if len(idx) == 0:
            return
        c = len(idx)
        xt = self.rng.uniform(-75, -3, (c, K))
        yt = self.rng.uniform(-37, 37, (c, K))
        xt[:, IS_G], yt[:, IS_G] = -86.0, 0.0
        self.pos[idx, :, 0] = SIGN_K * xt
        self.pos[idx, :, 1] = SIGN_K * yt
        self.vel[idx] = 0.0
        self.facing[idx] = SIGN_K[None, :, None] * np.array([1.0, 0.0])
        self._park(np.where(m[:, None], ~self.active, False))
        self._stoppage(m, np.zeros((len(m), 2)), SETUP_GOAL)

    def _stoppage(self, m, fo, setup):
        self.setup = np.where(m, setup, self.setup)
        self.fo_pt[m] = fo[m]
        self.ppos[m] = fo[m]
        self.pvel[m] = 0.0
        self.owner[m] = -1
        self.last_team[m] = -1
        self.icing_armed[m] = -1
        self.shot_live[m] = False
        self.pass_team[m] = -1
        self.cool[m] = 0.0

    # ------------------------------------------------------------ coach
    def _apply_coach(self, want):
        n_idx = np.arange(self.n)
        for T in (0, 1):
            g, x = T * E + GOALIE, T * E + EXTRA
            pull = want[:, T] & ~self.pulled[:, T]
            if pull.any():
                diff = self.score[:, T] - self.score[:, 1 - T]
                tl = 1.0 - self.t / self.T_steps
                for i in np.where(pull)[0]:
                    self.pull_log.append((tl, int(diff[i])))
                self.stats["pulls"][pull, T] += 1
                self.pulled[pull, T] = True
                self.active[pull, g] = False
                self._park(np.where(pull[:, None], np.arange(K)[None] == g, False))
                self.active[pull, x] = True
                self.pos[pull, x] = np.stack([np.full(pull.sum(), SIGN[T] * -12.0),
                                              self.bench_sign[pull] * 40.0], -1)
                self.vel[pull, x] = 0.0
                self.facing[pull, x] = [SIGN[T], 0.0]
                self.cool[pull, x] = 0.0
            back = ~want[:, T] & self.pulled[:, T] & (self.setup > 0)
            if back.any():
                self.pulled[back, T] = False
                self.active[back, x] = False
                self._park(np.where(back[:, None], np.arange(K)[None] == x, False))
                self.active[back, g] = True
                self.pos[back, g] = [SIGN[T] * -86.0, 0.0]
                self.vel[back, g] = 0.0
        del n_idx

    # ------------------------------------------------------------ puck actions
    def _puck_actions(self, puck_act):
        live = (self.setup == 0) & (self.owner >= 0)
        if not live.any():
            return
        idx = np.where(live)[0]
        o = self.owner[idx]
        T = TEAM_OF[o]
        slot = SLOT_OF[o]
        a = np.where(slot < N_SLOTS, puck_act[idx, T, np.minimum(slot, N_SLOTS - 1)], 0)
        rel = a > 0
        if not rel.any():
            return
        idx, o, T, a = idx[rel], o[rel], T[rel], a[rel]
        s = SIGN[T][:, None]
        p = self.ppos[idx] * s                       # team frame
        c = len(idx)
        tgt = np.zeros((c, 2))
        speed = np.zeros(c)
        sigma = np.zeros(c)
        shoot = a == 1
        tgt[shoot] = np.stack([np.full(shoot.sum(), GOAL_X),
                               self.rng.uniform(-2.7, 2.7, shoot.sum())], -1)
        speed[shoot] = SHOT_SPEED * self.rng.uniform(0.9, 1.1, shoot.sum())
        sigma[shoot] = SHOT_SIGMA
        for code, side in ((2, 1.0), (3, -1.0)):
            m = a == code
            tgt[m] = [RINK_X - 5.0, side * (RINK_Y - 8.0)]
            speed[m], sigma[m] = DUMP_SPEED, DUMP_SIGMA
        pas = a >= 4
        if pas.any():
            j = T[pas] * E + (a[pas] - 4)
            tp = self.pos[idx[pas], j] * s[pas]
            tv = self.vel[idx[pas], j] * s[pas]
            lead = _norm(tp - p[pas]) / PASS_SPEED
            tgt[pas] = tp + tv * lead[:, None]
            speed[pas], sigma[pas] = PASS_SPEED, PASS_SIGMA
            self.stats["passes"][idx[pas], T[pas]] += 1
            self.pass_team[idx[pas]] = T[pas]
        self.stats["shots"][idx[shoot], T[shoot]] += 1
        d = tgt - p
        d = d / (_norm(d)[:, None] + 1e-9)
        d = _rot(d, self.rng.normal(0, 1, c) * sigma)
        self.pvel[idx] = d * speed[:, None] * s
        self.owner[idx] = -1
        self.last_team[idx] = T
        self.cool[idx, o] = COOLDOWN
        self.shot_live[idx] = shoot
        self.pass_team[idx[~pas]] = -1
        self.icing_armed[idx] = np.where(p[:, 0] < 0, T, -1) if self.icing_on else -1

    # ------------------------------------------------------------ players
    def _move_players(self, move):
        act = self.active
        wdir = DIRS[move] * SIGN_K[None, :, None]
        v = self.vel + wdir * (ACC_K[None, :, None] * SUB_DT)
        sp = _norm(v)
        brake = move == 1
        dec = np.minimum(sp, BRAKE * SUB_DT)
        v = np.where(brake[..., None], v - v / (sp[..., None] + 1e-9) * dec[..., None], v)
        v *= (1 - DRAG * SUB_DT)
        vmax = np.broadcast_to(VMAX_K, move.shape).copy()
        has = self.owner >= 0
        vmax[np.where(has)[0], self.owner[has]] *= CARRY_SPEED
        sp = _norm(v)
        v *= np.minimum(1.0, vmax / (sp + 1e-9))[..., None]
        p = self.pos + v * SUB_DT
        moving = move >= 2
        self.facing = np.where(moving[..., None], wdir, self.facing)

        # player-player collisions (component-wise: much faster than [...,2] reductions)
        px, py, vx, vy = p[..., 0], p[..., 1], v[..., 0], v[..., 1]
        dx = px[:, :, None] - px[:, None, :]
        dy = py[:, :, None] - py[:, None, :]
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        both = act[:, :, None] & act[:, None, :] & ~np.eye(K, dtype=bool)[None]
        ov = np.clip(RAD_K[:, None] + RAD_K[None, :] - dist, 0, None) * both
        nx, ny = dx / dist, dy / dist
        rvx = vx[:, :, None] - vx[:, None, :]
        rvy = vy[:, :, None] - vy[:, None, :]
        vn = rvx * nx + rvy * ny
        touching = ov > 0
        imp = np.where(touching & (vn < 0), vn, 0.0) * (0.5 * (1 + PLAYER_REST))
        p = np.stack([px + 0.5 * (ov * nx).sum(2), py + 0.5 * (ov * ny).sum(2)], -1)
        v = np.stack([vx - (imp * nx).sum(2), vy - (imp * ny).sum(2)], -1)
        opp = TEAM_OF[:, None] != TEAM_OF[None, :]
        self.hit = (touching & (-vn > HIT_SPEED) & opp[None]).any(2)

        p, v = clamp_rink(p, v, RAD_K[None, :], PLAYER_REST)
        # nets are solid
        for sgn in (1.0, -1.0):
            x0, x1 = _net_box(sgn)
            q = np.stack([np.clip(p[..., 0], x0, x1), np.clip(p[..., 1], -POST_Y, POST_Y)], -1)
            dd = p - q
            dl = _norm(dd)
            inside = dl < 1e-6
            if inside.any():  # centre inside the frame: shove out the front
                p[..., 0] = np.where(inside, sgn * (GOAL_X - RAD_K[None, :]), p[..., 0])
            pen = (~inside) & (dl < RAD_K[None, :])
            if pen.any():
                nn = dd / (dl[..., None] + 1e-9)
                p = np.where(pen[..., None], q + nn * RAD_K[None, :, None], p)
                vv = _dot(v, nn)
                v = np.where((pen & (vv < 0))[..., None], v - vv[..., None] * nn, v)
        # goalies stay in front of their own net
        gx = SIGN_K[None, IS_G] * p[:, IS_G, 0]
        gy = SIGN_K[None, IS_G] * p[:, IS_G, 1]
        gx = np.maximum(gx, GOALIE_MIN_X)
        dx, dy = gx + GOAL_X, gy
        dg = np.sqrt(dx * dx + dy * dy) + 1e-9
        f = np.minimum(1.0, GOALIE_RANGE / dg)
        gx, gy = np.maximum(-GOAL_X + dx * f, GOALIE_MIN_X), dy * f
        moved = f < 1.0
        p[:, IS_G, 0] = SIGN_K[None, IS_G] * gx
        p[:, IS_G, 1] = SIGN_K[None, IS_G] * gy
        if moved.any():  # drop the outward part of the goalie's velocity
            nx, ny = dx / dg, dy / dg
            vg = v[:, IS_G]
            vo = (vg[..., 0] * nx + vg[..., 1] * ny) * SIGN_K[None, IS_G]
            vo = np.where(moved & (vo > 0), vo, 0.0)
            vg[..., 0] -= vo * nx * SIGN_K[None, IS_G]
            vg[..., 1] -= vo * ny * SIGN_K[None, IS_G]
            v[:, IS_G] = vg
        xt = SIGN_K * p[..., 0]
        lim = np.full(K, np.inf)
        over = xt > lim
        if self.setup.any():   # faceoff set-up: everyone on own side of the dot
            fo_t = SIGN_K[None, :] * self.fo_pt[:, None, 0] - 1.5
            lim2 = np.where((self.setup > 0)[:, None], fo_t, np.inf)
            lim = np.minimum(lim[None, :], lim2)
            over = xt > lim
        if over.any():
            p[..., 0] = np.where(over, SIGN_K * np.broadcast_to(lim, xt.shape), p[..., 0])
            v[..., 0] = np.where(over & (SIGN_K * v[..., 0] > 0), 0.0, v[..., 0])
        p[~act] = [0.0, 500.0]
        v[~act] = 0.0
        self.pos, self.vel = p, v

    # ------------------------------------------------------------ puck
    def _puck_substep(self, ev, ev_team):
        n = self.n
        idx = np.arange(n)
        live = (self.setup == 0) & (ev == EV_NONE)
        if not live.any():
            return
        self.cool = np.maximum(self.cool - SUB_DT, 0.0)
        p0 = self.ppos.copy()
        o = np.maximum(self.owner, 0)
        owned = live & (self.owner >= 0)

        # --- takeaways: stick checks and big hits
        if owned.any():
            opp = TEAM_OF[None, :] != TEAM_OF[o][:, None]
            dpk = _norm(self.pos - p0[:, None, :])
            close = opp & self.active & (dpk < STEAL_R) & owned[:, None]
            p_steal = 1 - (1 - STEAL_P) ** close.sum(1)
            steal = owned & (self.rng.random(n) < p_steal)
            knocked = owned & ~steal & self.hit[idx, o] & (self.rng.random(n) < 0.5)
            j = np.argmin(np.where(close, dpk, np.inf), 1)
            take = steal & (self.rng.random(n) < 0.5) & ~IS_G[j]
            loose_now = (steal & ~take) | knocked
            lost = steal | knocked
            if lost.any():
                self.cool[idx[lost], o[lost]] = 0.4
                ang = self.rng.uniform(0, 2 * np.pi, n)
                kick = np.stack([np.cos(ang), np.sin(ang)], -1) * 12.0
                self.pvel = np.where(loose_now[:, None], self.vel[idx, o] * 0.5 + kick, self.pvel)
                self.owner = np.where(take, j, np.where(loose_now, -1, self.owner))
                self.last_team = np.where(steal, TEAM_OF[j], self.last_team)
                tk = np.where(steal)[0]
                self.stats["takeaways"][tk, TEAM_OF[j[tk]]] += 1
                self.pass_team[lost] = -1
            o = np.maximum(self.owner, 0)
            owned = live & (self.owner >= 0)
        loose = live & (self.owner < 0)

        p1 = p0.copy()
        if owned.any():
            car = self.pos[idx, o] + self.facing[idx, o] * PUCK_OFFSET
            p1 = np.where(owned[:, None], car, p1)
            self.pvel = np.where(owned[:, None], self.vel[idx, o], self.pvel)
        if loose.any():
            sp = _norm(self.pvel)
            f = np.maximum(sp - PUCK_FRICTION * SUB_DT, 0) / (sp + 1e-9)
            f = np.where(sp < 0.5, 0.0, f)
            self.pvel = np.where(loose[:, None], self.pvel * f[:, None], self.pvel)
            p1 = np.where(loose[:, None], p0 + self.pvel * SUB_DT, p1)

        # --- loose puck meets a stick / body / goalie
        if loose.any():
            seg = p1 - p0
            sl2 = _dot(seg, seg) + 1e-9
            rel = self.pos - p0[:, None]
            t = np.clip(_dot(rel, seg[:, None]) / sl2[:, None], 0, 1)
            q = p0[:, None] + t[..., None] * seg[:, None]
            dq = _norm(self.pos - q)
            cand = loose[:, None] & self.active & (self.cool <= 0) & (dq < REACH_K[None])
            score = np.where(cand, t + 1e-3 * dq, np.inf)
            k = np.argmin(score, 1)
            hit = np.isfinite(score[idx, k])
            if hit.any():
                self._contact(np.where(hit)[0], k[hit], q[idx, k][hit], dq[idx, k][hit], p1, ev, ev_team)
            loose = live & (self.owner < 0) & (ev == EV_NONE)
            owned = live & (self.owner >= 0)

        # --- goal line
        for sgn, scorer in ((1.0, 0), (-1.0, 1)):
            gx = sgn * GOAL_X
            crossed = (sgn * p0[:, 0] < GOAL_X) & (sgn * p1[:, 0] >= GOAL_X)
            tc = (gx - p0[:, 0]) / (p1[:, 0] - p0[:, 0] + 1e-9)
            yc = p0[:, 1] + tc * (p1[:, 1] - p0[:, 1])
            goal = live & (ev == EV_NONE) & crossed & (np.abs(yc) < POST_Y)
            if goal.any():
                ev[goal], ev_team[goal] = EV_GOAL, scorer
                self.stats["goals"][goal, scorer] += 1
                self.stats["sog"][goal & self.shot_live, scorer] += 1
                p1[goal] = [sgn * (GOAL_X + 1.0), 0.0]

        # --- net frame and boards
        act = live & (ev == EV_NONE)
        for sgn in (1.0, -1.0):
            x0, x1 = _net_box(sgn)
            inside = act & (p1[:, 0] > x0) & (p1[:, 0] < x1) & (np.abs(p1[:, 1]) < POST_Y)
            if not inside.any():
                continue
            back = inside & (sgn * p0[:, 0] >= GOAL_X + NET_DEPTH)
            side = inside & ~back & (np.abs(p0[:, 1]) >= POST_Y)
            front = inside & ~back & ~side
            p1[:, 0] = np.where(back, sgn * (GOAL_X + NET_DEPTH + 0.05), p1[:, 0])
            p1[:, 1] = np.where(side, np.sign(p0[:, 1]) * (POST_Y + 0.05), p1[:, 1])
            p1[:, 0] = np.where(front, sgn * (GOAL_X - 0.05), p1[:, 0])
            lo = inside & (self.owner < 0)
            self.pvel[:, 0] = np.where(lo & (back | front), -PUCK_REST * self.pvel[:, 0], self.pvel[:, 0])
            self.pvel[:, 1] = np.where(lo & side, -PUCK_REST * self.pvel[:, 1], self.pvel[:, 1])
        pv = self.pvel.copy()
        p1, pv = clamp_rink(p1, pv, 0.1, PUCK_REST)
        self.pvel = np.where((self.owner < 0)[:, None], pv, self.pvel)

        # --- icing
        if self.icing_on:
            ic = act & (self.owner < 0) & (self.icing_armed >= 0)
            if ic.any():
                s = SIGN[np.maximum(self.icing_armed, 0)]
                ice = ic & (s * p0[:, 0] < GOAL_X) & (s * p1[:, 0] >= GOAL_X)
                if ice.any():
                    ev[ice], ev_team[ice] = EV_ICING, self.icing_armed[ice]
                    self.stats["icing"][ice, self.icing_armed[ice]] += 1
        # --- offside
        if self.offside_on:
            for T in (0, 1):
                x0t, x1t = SIGN[T] * p0[:, 0], SIGN[T] * p1[:, 0]
                entered = live & (ev == EV_NONE) & (x0t <= BLUE_X) & (x1t > BLUE_X)
                if not entered.any():
                    continue
                o = np.maximum(self.owner, 0)
                own_team = np.where(self.owner >= 0, TEAM_OF[o], self.last_team)
                entered &= own_team == T
                sk = slice(T * E, T * E + N_SLOTS)
                ahead = (SIGN[T] * self.pos[:, sk, 0] > BLUE_X + 1.0) & self.active[:, sk]
                ahead &= (np.arange(T * E, T * E + N_SLOTS)[None] != self.owner[:, None])
                off = entered & ahead.any(1)
                if off.any():
                    ev[off], ev_team[off] = EV_OFFSIDE, T
                    self.stats["offside"][off, T] += 1
        self.ppos = np.where(live[:, None], p1, self.ppos)

    def _contact(self, gi, k, q, dq, p1, ev, ev_team):
        """Resolve the first stick/body the loose puck meets in this sub-step."""
        sp = _norm(self.pvel[gi])
        T = TEAM_OF[k]
        is_g = IS_G[k]
        u = self.rng.random(len(gi))
        # goalie
        if is_g.any():
            g = is_g
            gg = gi[g]
            hard = sp[g] > 40
            saved = ~hard | (u[g] < SAVE_P)
            shot = self.shot_live[gg] & hard
            self.stats["sog"][gg[shot & saved], 1 - T[g][shot & saved]] += 1
            self.stats["saves"][gg[shot & saved], T[g][shot & saved]] += 1
            near = _norm(self.pos[gg, k[g]] - np.stack([-SIGN[T[g]] * GOAL_X, np.zeros(g.sum())], -1)) < FREEZE_DIST
            frz = saved & near & (self.rng.random(len(gg)) < FREEZE_P)
            reb = saved & ~frz
            if frz.any():
                ev[gg[frz]], ev_team[gg[frz]] = EV_FREEZE, T[g][frz]
                self.stats["freeze"][gg[frz], T[g][frz]] += 1
                p1[gg[frz]] = q[g][frz]
            if reb.any():
                r = gg[reb]
                nrm = q[g][reb] - self.pos[r, k[g][reb]]
                nrm = nrm / (_norm(nrm)[:, None] + 1e-9)
                bad = _norm(nrm) < 0.5
                nrm[bad] = np.stack([SIGN[T[g][reb][bad]], np.zeros(bad.sum())], -1)
                v = self.pvel[r]
                vn = _dot(v, nrm)
                v = v - 2 * np.minimum(vn, 0)[:, None] * nrm
                v = v / (_norm(v)[:, None] + 1e-9)
                v = _rot(v, self.rng.uniform(-0.7, 0.7, len(r))) * self.rng.uniform(10, 30, len(r))[:, None]
                self.pvel[r] = v
                p1[r] = q[g][reb]
                self.last_team[r] = T[g][reb]
                self.icing_armed[r] = -1
                self.shot_live[r] = False
                self.pass_team[r] = -1
            self.cool[gg, k[g]] = 0.25
        # skaters
        s = ~is_g
        if s.any():
            gs, ks, Ts = gi[s], k[s], T[s]
            cap_p = np.where(sp[s] < 45, 1.0, np.maximum(0.3, 1 - (sp[s] - 45) / 80))
            cap = u[s] < cap_p
            if cap.any():
                c = gs[cap]
                ok = self.pass_team[c] == Ts[cap]
                self.stats["pass_ok"][c[ok], Ts[cap][ok]] += 1
                self.owner[c] = ks[cap]
                self.last_team[c] = Ts[cap]
                self.icing_armed[c] = -1
                self.shot_live[c] = False
                self.pass_team[c] = -1
                p1[c] = self.pos[c, ks[cap]] + self.facing[c, ks[cap]] * PUCK_OFFSET
                self.pvel[c] = self.vel[c, ks[cap]]
            body = ~cap & (dq[s] < RAD_K[ks])
            if body.any():
                b = gs[body]
                v = _rot(self.pvel[b], self.rng.uniform(-1.0, 1.0, len(b))) * 0.5
                self.pvel[b] = v
                p1[b] = q[s][body]
                self.last_team[b] = Ts[body]
                self.icing_armed[b] = -1
                self.shot_live[b] = False
                self.pass_team[b] = -1
            self.cool[gs[~cap], ks[~cap]] = 0.15

    # ------------------------------------------------------------ step
    def step(self, move, puck_act, want_pull):
        """move: int [N,2,7] team-frame move action for every entity
        puck_act: int [N,2,6] skater puck action
        want_pull: bool [N,2] coach wants the goalie out
        returns obs, reward [N,2] (goals for - against this step), done, events [N]"""
        n = self.n
        self._apply_coach(want_pull)
        self._puck_actions(puck_act)
        mv = move.reshape(n, K)
        ev = np.zeros(n, int)
        ev_team = np.zeros(n, int)
        for _ in range(SUBSTEPS):
            self._move_players(mv)
            self._puck_substep(ev, ev_team)

        rew = np.zeros((n, 2))
        h = self.owner >= 0
        self.stats["poss_seconds"][np.where(h)[0], TEAM_OF[self.owner[h]]] += DT
        # set-up countdown and puck drop
        was = self.setup > 0
        self.setup = np.maximum(self.setup - 1, 0)
        drop = was & (self.setup == 0)
        if drop.any():
            ang = self.rng.uniform(0, 2 * np.pi, n)
            v = np.stack([np.cos(ang), np.sin(ang)], -1) * 3.0
            self.pvel = np.where(drop[:, None], v, self.pvel)
        # whistles
        g = ev == EV_GOAL
        if g.any():
            sc = ev_team
            rew[g, sc[g]] += 1
            rew[g, 1 - sc[g]] -= 1
            self.score[g, sc[g]] += 1
            en = g & self.pulled[np.arange(n), 1 - sc]
            self.stats["en_goals_against"][en, 1 - sc[en]] += 1
            sx = g & self.pulled[np.arange(n), sc] & ~self.pulled[np.arange(n), 1 - sc]
            self.stats["goals_6v5"][sx, sc[sx]] += 1
            self._kickoff(g)
        oth = (ev != EV_NONE) & ~g
        if oth.any():
            side = np.where(self.ppos[:, 1] >= 0, 1.0, -1.0)
            fo = np.zeros((n, 2))
            s = SIGN[ev_team]
            fx = np.where(ev == EV_OFFSIDE, s * NEUTRAL_DOT[0], -s * END_DOT[0])
            fy = np.where(ev == EV_OFFSIDE, NEUTRAL_DOT[1], END_DOT[1]) * side
            fo[:, 0], fo[:, 1] = fx, fy
            self._stoppage(oth, fo, SETUP_STOP)
        self.t += 1
        done = self.t >= self.T_steps
        return self.obs(), rew, done, ev

    # ------------------------------------------------------------ observations
    def obs(self):
        n = self.n
        has = np.zeros((n, K))
        h = self.owner >= 0
        has[np.where(h)[0], self.owner[h]] = 1.0
        o_out = {}
        sgn = SIGN[None, :, None]                               # [1,2,1]
        P = self.pos[:, VIEW] * sgn[..., None]                  # [N,2,14,2]
        V = self.vel[:, VIEW] * sgn[..., None]
        A = self.active[:, VIEW].astype(np.float64)             # [N,2,14]
        Hp = has[:, VIEW]
        scale = np.array([RINK_X, RINK_Y])
        ent = np.concatenate([A[..., None], P / scale, V / 30.0, Hp[..., None]], -1) * A[..., None]
        pp = self.ppos[:, None, :] * sgn                        # [N,2,2]
        pv = self.pvel[:, None, :] * sgn
        own_t = np.where(h, TEAM_OF[np.maximum(self.owner, 0)], -1)
        poss = np.zeros((n, 2, 3))
        for T in (0, 1):
            poss[:, T, 0] = own_t < 0
            poss[:, T, 1] = own_t == T
            poss[:, T, 2] = own_t == 1 - T
        puck = np.concatenate([pp / scale, pv / 60.0, poss], -1)
        diff = (self.score[:, [0, 1]] - self.score[:, [1, 0]]).astype(np.float64)
        tl = 1.0 - self.t / self.T_steps
        fo = self.fo_pt[:, None, :] * sgn
        game = np.stack([
            np.clip(diff, -3, 3) / 3.0,
            np.full((n, 2), tl),
            np.repeat((self.setup > 0)[:, None], 2, 1).astype(float),
            np.repeat(self.setup[:, None] / SETUP_GOAL, 2, 1),
            fo[..., 0] / RINK_X, fo[..., 1] / RINK_Y,
            self.pulled[:, [0, 1]].astype(float), self.pulled[:, [1, 0]].astype(float),
            np.stack([self.icing_armed == 0, self.icing_armed == 1], -1).astype(float),
            np.stack([self.icing_armed == 1, self.icing_armed == 0], -1).astype(float),
            (pp[..., 0] > BLUE_X).astype(float), (pp[..., 0] < -BLUE_X).astype(float),
        ], -1)
        team = np.concatenate([ent.reshape(n, 2, -1), puck, game], -1)   # [N,2,103]

        Ps, Vs = P[:, :, :E], V[:, :, :E]                        # own entities = agents
        rel_p = (pp[:, :, None, :] - Ps) / 50.0
        agent = np.concatenate([
            np.broadcast_to(np.eye(E), (n, 2, E, E)),
            Ps / scale, Vs / 30.0, Hp[:, :, :E, None], rel_p,
            _norm(pp[:, :, None, :] - Ps)[..., None] / 100.0,
            (self.cool[:, VIEW][:, :, :E] > 0)[..., None].astype(float),
        ], -1) * A[:, :, :E, None]
        rel = (P[:, :, None, :, :] - Ps[:, :, :, None, :]) / 50.0          # [N,2,7,14,2]
        rel = rel * A[:, :, None, :, None] * A[:, :, :E, None, None]
        full = np.concatenate([np.broadcast_to(team[:, :, None], (n, 2, E, team.shape[-1])),
                               agent, rel.reshape(n, 2, E, -1)], -1).astype(np.float32)
        o_out["obs"] = full                                      # [N,2,7,OBS_DIM]
        o_out["active"] = self.active.reshape(n, 2, E).copy()

        mask = np.zeros((n, 2, N_SLOTS, N_PUCK), bool)
        mask[..., 0] = True
        live_own = h & (self.setup == 0)
        if live_own.any():
            ii = np.where(live_own)[0]
            ok = self.owner[ii]
            T, sl = TEAM_OF[ok], SLOT_OF[ok]
            sk = sl < N_SLOTS
            ii, T, sl = ii[sk], T[sk], sl[sk]
            mask[ii, T, sl, 1:4] = True
            mates = self.active.reshape(n, 2, E)[ii, T, :N_SLOTS].copy()
            mates[np.arange(len(ii)), sl] = False
            mask[ii, T, sl, 4:] = mates
        o_out["puck_mask"] = mask

        oh = np.zeros((n, 2, 7))
        oh[np.arange(n)[:, None], np.arange(2)[None], (np.clip(diff, -3, 3) + 3).astype(int)] = 1
        coach = np.concatenate([
            oh, np.full((n, 2, 1), tl), np.full((n, 2, 1), tl * tl),
            self.pulled[:, [0, 1], None].astype(float), self.pulled[:, [1, 0], None].astype(float),
            poss, pp[..., :1] / RINK_X, np.repeat((self.setup > 0)[:, None, None], 2, 1).astype(float),
        ], -1).astype(np.float32)
        o_out["coach"] = coach
        return o_out

    # ------------------------------------------------------------ analysis helpers
    def team_frame_positions(self):
        """[N,2,7,2] each team's entities in its own frame (attacking +x)."""
        return self.pos.reshape(self.n, 2, E, 2) * SIGN[None, :, None, None]

    def roles(self):
        """Role index per regular skater [N,2,5] (valid where 5v5 & live)."""
        p = self.team_frame_positions()[:, :, :5]
        return role_assign(p[..., 0], p[..., 1])
