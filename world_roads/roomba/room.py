"""The cold open: a first-generation Roomba in a living room. No map: it drives straight until it bumps into
something, turns a random way and carries on. Given time, it still covers nearly the whole floor.

A simulation, not footage: 0.3 m/s, a 34 cm robot, a 6 x 4.2 m room with furniture.
"""
import math
import numpy as np
import cv2

PX_M = 180.0                                   # pixels per metre
RW, RH = 6.0, 4.2                              # room, metres
R = 0.17                                       # Roomba radius
V = 0.3                                        # m/s
# furniture: (x0, y0, x1, y1) in metres from the room's top-left; y down
BOXES = [("sofa", (1.6, 0.0, 3.9, 0.9)), ("coffee table", (2.3, 1.75, 3.4, 2.35)), ("armchair", (5.0, 1.6, 6.0, 2.6)),
         ("TV stand", (1.8, 3.75, 3.7, 4.2)), ("shelf", (0.0, 1.2, 0.4, 2.8))]
PLANT = (0.55, 3.65, 0.3)                      # a round pot: x, y, radius


def blocked(x, y):
    """is the robot centre at (x, y) touching anything?"""
    if x < R or y < R or x > RW - R or y > RH - R:
        return True
    for _, (x0, y0, x1, y1) in BOXES:
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        if dx * dx + dy * dy < R * R:
            return True
    return math.hypot(x - PLANT[0], y - PLANT[1]) < R + PLANT[2]


def simulate(seconds=2700.0, dt=0.02, seed=3):
    """positions every dt; turns at random after each bump (away from the obstacle)"""
    rng = np.random.default_rng(seed)
    x, y, th = 1.0, 2.9, -0.35
    out = np.empty((int(seconds / dt) + 1, 2))
    out[0] = x, y
    for i in range(1, len(out)):
        nx, ny = x + V * dt * math.cos(th), y + V * dt * math.sin(th)
        if blocked(nx, ny):
            # find a free direction: random headings until one is clear a little way ahead
            for _ in range(200):
                th = rng.uniform(-math.pi, math.pi)
                if not blocked(x + 0.05 * math.cos(th), y + 0.05 * math.sin(th)):
                    break
            nx, ny = x, y                     # the turn takes this step
        x, y = nx, ny
        out[i] = x, y
    return out


class Room:
    def __init__(self, ox, oy, w, h):
        self.ox, self.oy, self.w, self.h = ox, oy, w, h        # room's top-left on screen; canvas size
        self.dt = 0.02
        self.path = simulate(dt=self.dt)
        self.P = np.c_[ox + self.path[:, 0] * PX_M, oy + self.path[:, 1] * PX_M]
        self.cov = np.zeros((h, w), np.uint8)                 # swept floor
        self.k = 0
        self.free = np.zeros((h, w), np.uint8)                # floor the robot can reach (for the % cleaned)
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        mx, my = (X - ox) / PX_M, (Y - oy) / PX_M
        inside = (mx >= 0) & (mx <= RW) & (my >= 0) & (my <= RH)
        obst = np.zeros_like(inside)
        for _, (x0, y0, x1, y1) in BOXES:
            obst |= (mx >= x0) & (mx <= x1) & (my >= y0) & (my <= y1)
        obst |= np.hypot(mx - PLANT[0], my - PLANT[1]) <= PLANT[2]
        self.free = inside & ~obst
        self.nfree = int(self.free.sum())

    def advance(self, sim_t):
        k1 = min(int(sim_t / self.dt), len(self.P) - 1)
        if k1 > self.k:
            pts = self.P[self.k:k1 + 1].astype(np.int32)
            cv2.polylines(self.cov, [pts], False, 1, thickness=int(2 * R * PX_M * 0.92), lineType=cv2.LINE_8)
            self.k = k1
        return k1

    def covered(self):
        return float((self.cov.astype(bool) & self.free).sum()) / self.nfree

    def heading(self, k):
        j = max(0, k - 5)
        dx, dy = self.P[k] - self.P[j]
        return math.atan2(dy, dx) if dx * dx + dy * dy > 1e-6 else None
