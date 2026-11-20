"""Stage 1: the Moon breaks into seven pieces, and the pieces fly, collide and break.

REBOUND (IAS15) integrates the Sun, the Earth and every fragment above `m_cut` with full mutual
gravity. Collisions halt the integrator; `collisions.resolve` decides the outcome. Smaller debris is
handed to the tracer stage as test particles (`tracers.py`).

Output (out/<scenario>/):
  field.npz    geocentric states of the gravitating fragments (>= m_grav) and the Sun, sampled every
               `dt_sample`, plus duplicate samples at every collision so slot changes are clean
  bodies.npz   every N-body fragment every `dt_frame` (for pictures)
  tracers0.npz debris born in the breakup and in fragment collisions (birth time, state, mass)
  events.json  every collision and every fragment that hit the Earth or left
"""
import json
import warnings
import math
import os
import sys
import time

import numpy as np
import rebound

from constants import (G, M_SUN, M_EARTH, R_ENTRY, M_MOON, R_MOON, AU, DAY, YEAR, A_MOON, E_MOON,
                       I_MOON, E_EARTH, U_BIND_MOON, V_ESC_MOON, R_ESCAPE, radius_of)
import collisions

warnings.filterwarnings('ignore', message='Name hash table')

DEFAULTS = dict(
    seed=7,
    n_pieces=7,
    v_disperse=150.0,         # m/s: rms speed the pieces keep once they have climbed out of each other's
                              # gravity. Breakup KE = |W| + M v^2/2 (W = pieces' mutual potential energy)
    debris0=0.02,             # fraction of the Moon turned into small debris by the breakup itself
    debris0_v_median=0.25,    # its speed beyond the expansion, in units of the Moon's escape speed
    m_cut=1e-4 * M_MOON,      # fragments above this are N-body bodies (radius ~80 km)
    m_grav=1e-3 * M_MOON,     # fragments above this pull on the debris tracers (radius ~170 km)
    m_quantum=5e-6 * M_MOON,  # mass of one swarm tracer (3.7e17 kg)
    size_slope=0.85,          # cumulative N(>m) ~ m^-b for debris
    cstar=5.0, mubar=0.37,    # L&S 2012 values for gravity-dominated ("strengthless") planets
    v_body_median=0.3,        # launch speed at infinity of new big fragments / mutual escape speed
    v_debris_median=0.6,      # same for debris
    f_energy=0.5,             # debris may carry at most this fraction of the impact energy
    moon_f=2.0,               # the Moon's true anomaly at the breakup (sets the lunar phase)
    years=20.0,
    dt_sample=0.1 * DAY,
    dt_frame=1.0 * DAY,
    max_bodies=400,
)


def voronoi_pieces(n, rng, npts=300_000, full=False):
    """Split a uniform sphere into n power-diagram cells. Returns mass fractions and centroids (unit R);
    with full=True also the seeds and power weights that define the cells (for the film's meshes)."""
    p = rng.normal(size=(npts, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    p *= rng.random(npts)[:, None] ** (1 / 3)
    seeds = rng.normal(size=(n, 3))
    seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
    seeds *= (0.25 + 0.6 * rng.random(n))[:, None]
    w = 0.25 * rng.random(n)                       # power weights -> uneven sizes
    d2 = ((p[:, None, :] - seeds[None]) ** 2).sum(-1) - w[None]
    lab = d2.argmin(1)
    frac = np.bincount(lab, minlength=n) / npts
    cen = np.stack([p[lab == k].mean(0) for k in range(n)])
    if full:
        return frac, cen, seeds, w
    return frac, cen


class FragmentSim:
    def __init__(self, cfg, outdir):
        self.cfg = cfg
        self.out = outdir
        self.rng = np.random.default_rng(cfg['seed'])
        self.events = []
        self.next_id = 1
        self.tracers = dict(t=[], x=[], v=[], m=[], src=[])
        self.slot_of = {}       # body id -> slot for gravitating bodies
        self.free_slots = []
        self.n_slots = 0

    # ---------- setup ----------
    def build(self):
        cfg, rng = self.cfg, self.rng
        sim = rebound.Simulation()
        sim.G = G
        sim.integrator = 'ias15'
        sim.collision = 'line'
        self.col_pairs = []

        def on_collision(simp, c):              # record the pair and halt (status 7 = collision)
            self.col_pairs.append((c.p1, c.p2))
            simp.contents._status = 7
            return 0
        self._on_collision = on_collision
        sim.collision_resolve = on_collision
        sim.add(m=M_SUN, name='sun')
        sim.add(m=M_EARTH, r=R_ENTRY, a=AU, e=E_EARTH, primary=sim.particles[0], name='earth')
        moon = rebound.Particle(simulation=sim, primary=sim.particles[1], m=M_MOON, a=A_MOON, e=E_MOON,
                                inc=I_MOON, Omega=0.3, omega=1.1, f=cfg.get('moon_f', 2.0))
        xm = np.array(moon.xyz)
        vm = np.array(moon.vxyz)
        self.sim = sim

        frac, cen = voronoi_pieces(cfg['n_pieces'], rng)
        m_pieces = frac * M_MOON * (1 - cfg['debris0'])
        r = cen * R_MOON
        r -= (m_pieces[:, None] * r).sum(0) / m_pieces.sum()
        # potential energy between the pieces (point masses), and the expansion that beats it
        W = 0.0
        for i in range(len(m_pieces)):
            for j in range(i + 1, len(m_pieces)):
                W -= G * m_pieces[i] * m_pieces[j] / np.linalg.norm(r[i] - r[j])
        I2 = (m_pieces * (r ** 2).sum(1)).sum()
        ke = abs(W) + 0.5 * m_pieces.sum() * cfg['v_disperse'] ** 2
        H = math.sqrt(2 * ke / I2)
        # tidally locked spin about the orbit normal
        hvec = np.cross(xm - np.array(sim.particles[1].xyz), vm - np.array(sim.particles[1].vxyz))
        omega = 2 * math.pi / (27.32 * DAY) * hvec / np.linalg.norm(hvec)
        for mi, ri in zip(m_pieces, r):
            self.add_body(mi, xm + ri, vm + np.cross(omega, ri) + H * ri)
        self.breakup = dict(pieces_kg=m_pieces.tolist(), pieces_frac_of_moon=(m_pieces / M_MOON).tolist(),
                            radii_km=[radius_of(m) / 1e3 for m in m_pieces], W=W, H=H,
                            ke_expansion=ke, ke_over_moon_binding=ke / U_BIND_MOON,
                            expansion_speed_surface_kms=H * R_MOON / 1e3)
        # debris from the breakup itself
        md = cfg['debris0'] * M_MOON
        n = int(round(md / cfg['m_quantum']))
        d = collisions.random_dirs(n, rng)
        rr = R_MOON * rng.random(n) ** (1 / 3)
        pos = rr[:, None] * collisions.random_dirs(n, rng)
        extra = collisions.launch_speeds(n, V_ESC_MOON, cfg['debris0_v_median'], 0.6, rng, 3.0)
        vel = vm + np.cross(omega, pos) + H * pos + extra[:, None] * d
        self.add_tracers(0.0, xm + pos, vel, np.full(n, md / max(n, 1)), 'breakup')
        sim.move_to_com()

    def add_body(self, m, x, v):
        bid = self.next_id
        self.next_id += 1
        self.sim.add(m=m, r=radius_of(m), x=x[0], y=x[1], z=x[2], vx=v[0], vy=v[1], vz=v[2], name=str(bid))
        if m >= self.cfg['m_grav']:
            if self.free_slots:
                s = self.free_slots.pop()
            else:
                s = self.n_slots
                self.n_slots += 1
            self.slot_of[bid] = s
        return bid

    def remove_body(self, bid):
        if bid in self.slot_of:
            self.free_slots.append(self.slot_of.pop(bid))
        self.sim.remove(str(bid))

    def add_tracers(self, t, x, v, m, src):
        """x, v in the simulation frame; stored geocentric."""
        if len(m) == 0:
            return
        e = self.sim.particles[1]
        self.tracers['t'].append(np.full(len(m), t))
        self.tracers['x'].append(np.asarray(x) - np.array(e.xyz))
        self.tracers['v'].append(np.asarray(v) - np.array(e.vxyz))
        self.tracers['m'].append(np.asarray(m))
        self.tracers['src'].append(np.full(len(m), {'breakup': 0, 'collision': 1}[src], np.int8))

    # ---------- sampling ----------
    def sample(self):
        sim = self.sim
        e = sim.particles[1]
        ex, ev = np.array(e.xyz), np.array(e.vxyz)
        row = np.zeros((max(self.n_slots, 1), 8))
        for p in sim.particles[2:]:
            s = self.slot_of.get(int(p.name))
            if s is not None:
                row[s, :3] = np.array(p.xyz) - ex
                row[s, 3:6] = np.array(p.vxyz) - ev
                row[s, 6] = G * p.m
                row[s, 7] = p.r
        sun = sim.particles[0]
        self.T.append(sim.t)
        self.F.append(row)
        self.S.append(np.r_[np.array(sun.xyz) - ex, np.array(sun.vxyz) - ev])

    def frame(self):
        sim = self.sim
        e = sim.particles[1]
        ex = np.array(e.xyz)
        n = sim.N - 2
        a = np.zeros((n, 5))
        for k, p in enumerate(sim.particles[2:]):
            a[k, :3] = np.array(p.xyz) - ex
            a[k, 3] = p.m
            a[k, 4] = int(p.name)
        self.frames_t.append(sim.t)
        self.frames.append(a)

    # ---------- collisions ----------
    def handle_collisions(self):
        sim = self.sim
        t = sim.t
        e = sim.particles[1]
        ex, ev = np.array(e.xyz), np.array(e.vxyz)
        # copy everything out first: removing particles invalidates references into the array
        idx = sorted({i for pair in self.col_pairs for i in pair if i >= 2})
        self.col_pairs = []
        hit = [dict(h=int(sim.particles[i].name), m=sim.particles[i].m, r=sim.particles[i].r,
                    x=np.array(sim.particles[i].xyz), v=np.array(sim.particles[i].vxyz)) for i in idx]
        rest = []
        for p in hit:
            if np.linalg.norm(p['x'] - ex) < R_ENTRY + p['r'] + 1e3:
                v = np.linalg.norm(p['v'] - ev)
                self.events.append(dict(kind='earth_impact', t=t, m=p['m'], r_km=p['r'] / 1e3, v=v,
                                        energy=0.5 * p['m'] * v * v))
                self.remove_body(p['h'])
            else:
                rest.append(p)
        while len(rest) >= 2:
            p = rest.pop(0)
            d = [np.linalg.norm(p['x'] - q['x']) - p['r'] - q['r'] for q in rest]
            j = int(np.argmin(d))
            if d[j] > 0.05 * (p['r'] + rest[j]['r']):
                continue
            q = rest.pop(j)
            res = collisions.resolve(p['m'], p['x'], p['v'], q['m'], q['x'], q['v'], self.cfg, self.rng)
            self.remove_body(p['h'])
            self.remove_body(q['h'])
            new = []
            for m, x, v in res['bodies']:
                if m >= self.cfg['m_cut']:
                    new.append(self.add_body(m, x, v))
                else:   # small remnants become tracers
                    res['tr_m'] = np.r_[res['tr_m'], m]
                    res['tr_x'] = np.vstack([res['tr_x'], x])
                    res['tr_v'] = np.vstack([res['tr_v'], v])
            self.add_tracers(t, res['tr_x'], res['tr_v'], res['tr_m'], 'collision')
            info = res['info']
            info.update(kind='collision', t=t, ids=[p['h'], q['h']], new=new)
            self.events.append(info)

    def remove_escapers(self):
        sim = self.sim
        ex = np.array(sim.particles[1].xyz)
        gone = [(int(p.name), p.m) for p in sim.particles[2:] if np.linalg.norm(np.array(p.xyz) - ex) > R_ESCAPE]
        if gone:
            self.sample()
        for h, m in gone:
            self.events.append(dict(kind='escaped', t=sim.t, m=m))
            self.remove_body(h)
        return len(gone)

    def separated(self):
        ps = [(np.array(p.xyz), p.r) for p in self.sim.particles[2:]]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                if np.linalg.norm(ps[i][0] - ps[j][0]) < ps[i][1] + ps[j][1]:
                    return False
        return True

    # ---------- main loop ----------
    def run(self):
        cfg = self.cfg
        sim = self.sim
        t_end = cfg['years'] * YEAR
        self.T, self.F, self.S = [], [], []
        self.frames_t, self.frames = [], []
        self.sample()
        self.frame()
        k_frame = int(round(cfg['dt_frame'] / cfg['dt_sample']))
        n_steps = int(math.ceil(t_end / cfg['dt_sample']))
        wall = time.time()
        # the seven pieces start out overlapping (they are spheres standing in for irregular cells),
        # so collisions stay off until every pair has separated
        separating = True
        sim.collision = 'none'
        for k in range(1, n_steps + 1):
            t = k * cfg['dt_sample']
            while True:
                try:
                    sim.integrate(t, exact_finish_time=1)
                    break
                except rebound.Collision:
                    self.sample()           # pre-collision state at sim.t
                    self.handle_collisions()
                    self.sample()           # post-collision state at the same time
            self.remove_escapers()          # samples the pre-removal state itself
            if separating and self.separated():
                separating = False
                sim.collision = 'line'
                print(f'  pieces separated after {sim.t / 3600:.1f} h; collisions on', flush=True)
            self.sample()
            if k % k_frame == 0:
                self.frame()
            if k % int(YEAR / cfg['dt_sample']) == 0:
                nc = sum(e['kind'] == 'collision' for e in self.events)
                print(f'  year {t / YEAR:5.1f}: {sim.N - 2} fragments, {nc} collisions, '
                      f'{sum(len(a) for a in self.tracers["m"])} tracers, {time.time() - wall:.0f}s', flush=True)
            if sim.N - 2 > cfg['max_bodies']:
                print('  too many N-body fragments; raise m_cut', flush=True)
                break
        self.save()

    def save(self):
        os.makedirs(self.out, exist_ok=True)
        S = max(r.shape[0] for r in self.F)
        F = np.zeros((len(self.F), S, 8))
        for i, r in enumerate(self.F):
            F[i, :r.shape[0]] = r
        np.savez(os.path.join(self.out, 'field.npz'), T=np.array(self.T), F=F, SUN=np.array(self.S))
        nmax = max(len(a) for a in self.frames)
        B = np.full((len(self.frames), nmax, 5), np.nan, np.float32)
        for i, a in enumerate(self.frames):
            B[i, :len(a)] = a
        np.savez_compressed(os.path.join(self.out, 'bodies.npz'), t=np.array(self.frames_t), B=B)
        tr = {k: np.concatenate(v) if v else np.zeros(0) for k, v in self.tracers.items()}
        np.savez(os.path.join(self.out, 'tracers0.npz'), **tr)
        cfg = {k: v for k, v in self.cfg.items()}
        with open(os.path.join(self.out, 'events.json'), 'w') as f:
            json.dump(dict(config=cfg, breakup=self.breakup, events=self.events), f, indent=1, default=float)


def main(name, **over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', name)
    fs = FragmentSim(cfg, out)
    fs.build()
    print(f'{name}: breakup KE = {fs.breakup["ke_over_moon_binding"]:.2f} x Moon binding energy, '
          f'pieces {np.round(np.array(fs.breakup["pieces_frac_of_moon"]) * 100, 1)} % of the Moon')
    fs.run()
    return out


if __name__ == '__main__':
    kw = {}
    for a in sys.argv[2:]:
        k, v = a.split('=')
        kw[k] = type(DEFAULTS[k])(float(v)) if k in DEFAULTS else float(v)
    main(sys.argv[1], **kw)
