"""How each prop moves, as functions of real time t (seconds), shared by the physics, the renders and the compositor.

The 50 mph ball: thrown from far left, it comes into view falling (it has already dropped ~1 m), hits the road at
x = 0.8 m, bounces (restitution 0.4 for steel on asphalt, the friction of the hit takes 15% of its speed and sets it
spinning) and clips the cone standing on its line.

The blast of the 17,500 mph pass reaches each prop at t_hit (the ball's passage at its x plus the shock's travel); the
impulse per area falls off as 1/distance from the path, ~200 Pa s at 2 m (rough: a 1 MJ/m line blast; a few atmospheres
for a few milliseconds close in). Cones and drums get that as a kick and rigid.py does the rest; the sign's post bends
(far beyond its plastic limit); the parked car, 1.5 t, only lurches and rocks, and its windows blow in."""
import math
import replay_time as T

G = 9.81
R_BALL = 0.10
ROLL_V = 22.352
ROLL_X0, ROLL_T0 = -20.0, 0.0          # launched at x0 at t0 (shot time), far off-screen left
X_LAND = 0.8
E_BOUNCE, K_FRIC = 0.40, 0.85
T_LAND = (X_LAND - ROLL_X0) / ROLL_V
H0 = R_BALL + 0.5 * G * T_LAND ** 2
VZ1 = E_BOUNCE * G * T_LAND
VX1 = K_FRIC * ROLL_V


def ball(t):
    """position and spin angle (about +y) of the 50 mph ball at shot time t"""
    if t < T_LAND:
        x = ROLL_X0 + ROLL_V * t
        z = H0 - 0.5 * G * t * t
        ang = 0.0
    else:
        u = t - T_LAND
        x = X_LAND + VX1 * u
        z = R_BALL + VZ1 * u - 0.5 * G * u * u
        if z < R_BALL:                       # second landing, off-screen: keep it rolling on
            z = R_BALL
        ang = VX1 / R_BALL * u
    return (x, T.PY, z), ang


# ---------------------------------------------------------------- the blast
I_2M = 200.0                           # Pa s at 2 m from the path


def impulse(d):
    return min(4 * I_2M, I_2M * 2.0 / max(d, 0.5))


def t_hit(x, d):
    a = (d / T.BLAST) ** 2 if d < T.R_WEAK else T.T_WEAK + (d - T.R_WEAK) / T.C_SOUND
    return (x - T.X_ENTER) / T.V + a


SIGN = dict(x=4.5, y=10.2, d=1.8, theta=math.radians(72))
CAR = dict(x=5.6, y=17.2, d=3.0,                       # the BMW: rear bumper 3 m beyond the path
           rear=(-1.62, -1.02, 0.96, 1.28), side=(-0.84, -0.95, 0.55, 0.96, 1.30))   # its rear and left glass, local m


def sign_bend(t):
    """angle the post leans toward the camera (radians), pivoting at its base"""
    th = t_hit(SIGN["x"], SIGN["d"])
    u = t - th
    if u <= 0: return 0.0
    s = 1 - math.exp(-u / 0.035) * (1 + u / 0.035)
    wob = math.radians(4) * math.exp(-u / 0.25) * math.sin(2 * math.pi * 7 * u)
    return SIGN["theta"] * s + wob * s


def car_motion(t):
    """(forward slide m, pitch rad nose-down positive, heave m) of the parked car"""
    th = t_hit(CAR["x"], CAR["d"])
    u = t - th
    if u <= 0: return 0.0, 0.0, 0.0
    slide = 0.035 * (1 - math.exp(-u / 0.12))
    w, z = 2 * math.pi * 1.6, 0.25
    pitch = math.radians(2.6) * math.exp(-z * w * u) * math.sin(w * u)
    heave = -0.03 * math.exp(-z * w * u) * math.sin(w * u + 0.6)
    return slide, pitch, heave


def car_glass_broken(t):
    return t >= t_hit(CAR["x"], CAR["d"])
