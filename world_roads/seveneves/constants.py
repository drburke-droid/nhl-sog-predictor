"""Physical constants (SI) used everywhere in the Seveneves simulator."""
import math

G = 6.67430e-11            # m^3 kg^-1 s^-2
M_SUN = 1.98847e30         # kg
M_EARTH = 5.9722e24        # kg
R_EARTH = 6.371e6          # m, mean radius
H_ENTRY = 100e3            # m, debris counts as "entered the atmosphere" below this altitude
R_ENTRY = R_EARTH + H_ENTRY
M_MOON = 7.342e22          # kg
R_MOON = 1.7374e6          # m
RHO_MOON = 3344.0          # kg/m^3, bulk density
AU = 1.495978707e11        # m
DAY = 86400.0
YEAR = 365.25 * DAY

A_MOON = 3.844e8           # m, semi-major axis of the lunar orbit
E_MOON = 0.0549
I_MOON = math.radians(5.145)   # to the ecliptic
E_EARTH = 0.0167

GM_EARTH = G * M_EARTH
GM_SUN = G * M_SUN
R_HILL_EARTH = AU * (M_EARTH / (3 * M_SUN)) ** (1 / 3)   # ~1.50e9 m
R_ESCAPE = 2.0 * R_HILL_EARTH                             # beyond this, debris has left Earth's sphere

# Earth and its oceans
A_EARTH = 4 * math.pi * R_EARTH ** 2                      # 5.10e14 m^2
M_OCEAN = 1.335e21         # kg (Charette & Smith 2010: 1.332e21 m^3 ... ~1.37e21 kg incl. salt; we use water mass)
SIGMA = 5.670374e-8        # Stefan-Boltzmann

# Moon's gravitational binding energy (uniform sphere) and escape speed
U_BIND_MOON = 3 * G * M_MOON ** 2 / (5 * R_MOON)          # 1.24e29 J
V_ESC_MOON = math.sqrt(2 * G * M_MOON / R_MOON)           # 2.38 km/s


def radius_of(m, rho=RHO_MOON):
    return (3 * m / (4 * math.pi * rho)) ** (1 / 3)
