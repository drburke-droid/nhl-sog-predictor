"""Position-based granular solver shared by simulate.py (conveyor dish) and
bowl_sim.py (round bowl).  Equal spheres, Coulomb friction, kinematic colliders."""
import numpy as np
from numba import njit


@njit(cache=True)
def build_grid(P, n, inv_h, ox, oy, oz, nx, ny, nz, cell_of, start, order):
    ncell = nx * ny * nz
    for c in range(ncell + 1):
        start[c] = 0
    for i in range(n):
        ix = int((P[i, 0] - ox) * inv_h)
        iy = int((P[i, 1] - oy) * inv_h)
        iz = int((P[i, 2] - oz) * inv_h)
        if ix < 0: ix = 0
        if iy < 0: iy = 0
        if iz < 0: iz = 0
        if ix >= nx: ix = nx - 1
        if iy >= ny: iy = ny - 1
        if iz >= nz: iz = nz - 1
        c = (iz * ny + iy) * nx + ix      # z most significant: bottom-up sweep
        cell_of[i] = c
        start[c + 1] += 1
    for c in range(ncell):
        start[c + 1] += start[c]
    fill = start[:ncell].copy()
    for i in range(n):
        c = cell_of[i]
        order[fill[c]] = i
        fill[c] += 1


@njit(cache=True)
def substep(P, Pprev, V, N, n_all, dt, g, d, mu, mu_wall, iters,
            shape, b0, b1, b2, b3, inv_h, ox, oy, oz, nx, ny, nz,
            cell_of, start, order, v_sleep, vmax):
    """One PBD step.  Container: shape 0 = rounded-rectangle dish with half
    sizes b0 x b1, corner radius b2, wall height b3; shape 1 = round bowl of
    radius b0 with a fillet of radius b1 between floor and wall."""
    r = 0.5 * d
    d2 = d * d
    # integrate
    for i in range(N):
        V[i, 2] -= g * dt
        sp = np.sqrt(V[i, 0] ** 2 + V[i, 1] ** 2 + V[i, 2] ** 2)
        if sp > vmax:
            f = vmax / sp
            V[i, 0] *= f; V[i, 1] *= f; V[i, 2] *= f
        for k in range(3):
            Pprev[i, k] = P[i, k]
            P[i, k] += V[i, k] * dt
    build_grid(P, n_all, inv_h, ox, oy, oz, nx, ny, nz, cell_of, start, order)

    for it in range(iters):
        for idx in range(n_all):
            i = order[idx]
            if i >= N:
                continue
            c = cell_of[i]
            cx = c % nx
            cy = (c // nx) % ny
            cz = c // (nx * ny)
            for dz in range(-1, 2):
                z2 = cz + dz
                if z2 < 0 or z2 >= nz:
                    continue
                for dy in range(-1, 2):
                    y2 = cy + dy
                    if y2 < 0 or y2 >= ny:
                        continue
                    for dx in range(-1, 2):
                        x2 = cx + dx
                        if x2 < 0 or x2 >= nx:
                            continue
                        c2 = (z2 * ny + y2) * nx + x2
                        for jj in range(start[c2], start[c2 + 1]):
                            j = order[jj]
                            if j <= i:
                                continue
                            ex = P[i, 0] - P[j, 0]
                            ey = P[i, 1] - P[j, 1]
                            ez = P[i, 2] - P[j, 2]
                            q = ex * ex + ey * ey + ez * ez
                            if q >= d2 or q < 1e-12:
                                continue
                            dist = np.sqrt(q)
                            pen = d - dist
                            nxn = ex / dist; nyn = ey / dist; nzn = ez / dist
                            kin = j >= N
                            wi = 1.0 if kin else 0.5
                            P[i, 0] += wi * pen * nxn
                            P[i, 1] += wi * pen * nyn
                            P[i, 2] += wi * pen * nzn
                            if not kin:
                                P[j, 0] -= 0.5 * pen * nxn
                                P[j, 1] -= 0.5 * pen * nyn
                                P[j, 2] -= 0.5 * pen * nzn
                            # friction on relative tangential displacement
                            rx = (P[i, 0] - Pprev[i, 0]) - (P[j, 0] - Pprev[j, 0])
                            ry = (P[i, 1] - Pprev[i, 1]) - (P[j, 1] - Pprev[j, 1])
                            rz = (P[i, 2] - Pprev[i, 2]) - (P[j, 2] - Pprev[j, 2])
                            rn = rx * nxn + ry * nyn + rz * nzn
                            tx = rx - rn * nxn; ty = ry - rn * nyn; tz = rz - rn * nzn
                            tl = np.sqrt(tx * tx + ty * ty + tz * tz)
                            if tl > 1e-12:
                                lim = mu * pen
                                s = 1.0 if tl < lim else lim / tl
                                P[i, 0] -= wi * s * tx
                                P[i, 1] -= wi * s * ty
                                P[i, 2] -= wi * s * tz
                                if not kin:
                                    P[j, 0] += 0.5 * s * tx
                                    P[j, 1] += 0.5 * s * ty
                                    P[j, 2] += 0.5 * s * tz
            # tray: floor and four walls (static, with friction)
            pen = r - P[i, 2]
            if pen > 0:
                P[i, 2] = r
                tx = P[i, 0] - Pprev[i, 0]; ty = P[i, 1] - Pprev[i, 1]
                tl = np.sqrt(tx * tx + ty * ty)
                if tl > 1e-12:
                    lim = mu_wall * pen
                    s = 1.0 if tl < lim else lim / tl
                    P[i, 0] -= s * tx; P[i, 1] -= s * ty
            if shape == 0:
                hx = b0; hy = b1; rc = b2
                if P[i, 2] < b3 + 4 * d:
                    if P[i, 0] > hx - r: P[i, 0] = hx - r
                    if P[i, 0] < -hx + r: P[i, 0] = -hx + r
                    if P[i, 1] > hy - r: P[i, 1] = hy - r
                    if P[i, 1] < -hy + r: P[i, 1] = -hy + r
                    ax = abs(P[i, 0]) - (hx - rc)
                    ay = abs(P[i, 1]) - (hy - rc)
                    if ax > 0 and ay > 0:          # rounded corner
                        q = np.sqrt(ax * ax + ay * ay)
                        lim = rc - r
                        if q > lim:
                            f = lim / q
                            sx = 1.0 if P[i, 0] > 0 else -1.0
                            sy = 1.0 if P[i, 1] > 0 else -1.0
                            P[i, 0] = sx * (hx - rc + ax * f)
                            P[i, 1] = sy * (hy - rc + ay * f)
            else:
                R = b0; rf = b1
                rr = np.sqrt(P[i, 0] * P[i, 0] + P[i, 1] * P[i, 1])
                if rr > 1e-9:
                    rn = rr
                    if rr > R - rf and P[i, 2] < rf:   # fillet between floor and wall
                        dxr = rr - (R - rf)
                        dzr = rf - P[i, 2]
                        q = np.sqrt(dxr * dxr + dzr * dzr)
                        lim = rf - r
                        if q > lim:
                            f = lim / q
                            rn = R - rf + dxr * f
                            P[i, 2] = rf - dzr * f
                    if rn > R - r:                    # wall
                        rn = R - r
                    if rn != rr:
                        f = rn / rr
                        P[i, 0] *= f; P[i, 1] *= f

    inv_dt = 1.0 / dt
    vs2 = v_sleep * v_sleep
    for i in range(N):
        vx = (P[i, 0] - Pprev[i, 0]) * inv_dt
        vy = (P[i, 1] - Pprev[i, 1]) * inv_dt
        vz = (P[i, 2] - Pprev[i, 2]) * inv_dt
        if vx * vx + vy * vy + vz * vz < vs2:
            V[i, 0] = 0.0; V[i, 1] = 0.0; V[i, 2] = 0.0
            P[i, 0] = Pprev[i, 0]; P[i, 1] = Pprev[i, 1]; P[i, 2] = Pprev[i, 2]
        else:
            V[i, 0] = vx * 0.999; V[i, 1] = vy * 0.999; V[i, 2] = vz * 0.999
