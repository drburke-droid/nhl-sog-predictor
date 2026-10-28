"""The plate's camera, solved from the road markings (measure.py) on the 1920x804 plate.

Lines on the ground parallel to the road meet at the vanishing point (939.2, 372.9), the horizon. In a pinhole camera a
ground line at lateral offset X runs with slope dx/dy = X / H, whatever the focal length, so the edge lines' slopes
(-2.905, +2.425) and an assumed 7.3 m between them (two 3.65 m lanes) give the camera height: 1.37 m. The focal length
only changes perspective within an object, not where things sit or how fast they cross (both go as y - horizon), so
it comes from the lens: f = 1250 px on the plate (~24 mm equivalent, a drone's), which makes the centre-line dashes
3.2 m long with 6.6 m gaps. Pitch and yaw follow from the vanishing point; roll is taken as 0 (the sea horizon on
the left is level to 0.2 degrees).
World: road centre line along +Y, X to the right, Z up, ground at Z = 0."""
import json, math, numpy as np
W, H_IMG = 1920, 804
CX, CY = 957.1, 399.9                      # principal point: centre of the original frame, in plate pixels
VP = (939.164, 372.902)
SLOPE_L, SLOPE_R, SLOPE_C = -2.90521, 2.42456, 0.03092
ROAD_WIDTH = 7.3
F = 1250.0
H = ROAD_WIDTH / (SLOPE_R - SLOPE_L)
PITCH = math.atan((CY - VP[1]) / F)        # down
YAW = math.atan((VP[0] - CX) / F)          # road direction, relative to the optical axis (negative = left)
CAM_X = -SLOPE_C * H                       # camera sits this far from the centre line
X_LEFT, X_RIGHT = CAM_X + SLOPE_L * H, CAM_X + SLOPE_R * H

def R():
    """world -> camera (x right, y down, z forward)"""
    a = YAW                                 # the road is seen this far from the optical axis (negative: left of it)
    f = np.array([-math.sin(a), math.cos(a), 0.0])
    r = np.array([math.cos(a), math.sin(a), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    fwd = f * math.cos(PITCH) - up * math.sin(PITCH)
    down = -(up * math.cos(PITCH) + f * math.sin(PITCH))
    return np.stack([r, down, fwd])

def project(P):
    P = np.atleast_2d(np.asarray(P, float)) - np.array([CAM_X, 0.0, H])
    c = P @ R().T
    return np.stack([CX + F * c[:, 0] / c[:, 2], CY + F * c[:, 1] / c[:, 2]], axis=1), c[:, 2]

def ground_at(x, y):
    """the ground point seen at plate pixel (x, y)"""
    ray = np.linalg.inv(R()) @ np.array([(x - CX) / F, (y - CY) / F, 1.0])
    t = -H / ray[2]
    return np.array([CAM_X, 0.0, H]) + t * ray

def blender():
    """camera settings for Blender: 36 mm sensor, horizontal fit"""
    return dict(lens=F * 36 / W, sensor=36.0, loc=(CAM_X, 0.0, H),
                rot_deg=(90 - math.degrees(PITCH), 0.0, math.degrees(YAW)), res=(W, H_IMG),
                shift_x=(CX - W / 2) / W, shift_y=-(CY - H_IMG / 2) / W)

if __name__ == "__main__":
    print(f"height {H:.3f} m, pitch {math.degrees(PITCH):.2f} deg down, yaw {math.degrees(YAW):.2f} deg, camera x {CAM_X:.3f}")
    print(f"edge lines at x = {X_LEFT:.2f} and {X_RIGHT:.2f}")
    for y in (804, 700, 600, 516, 470, 430):
        print(f"row {y}: ground {ground_at(CX, y).round(2)}")
    print(json.dumps(blender()))
