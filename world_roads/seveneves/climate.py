"""Stage 3: what the falling Moon does to the Earth.

Global energy-balance model, one column averaged over the whole planet:

  dH/dt = (1 - f_space) * P_rain / A  +  S_abs  -  OLR(T)  -  k (T - T_deep)
  C_deep dT_deep/dt = k (T - T_deep), capped at F_MIX_MAX = 2 W/m^2 (a hot surface layer over cold
  water is so stable that tides and wind cannot mix more heat down)

H(T) is the enthalpy of the surface system relative to today: air, the ocean mixed layer, and all the
water that has evaporated into the sky at T (water vapour column = p_sat(T)/g, from the IAPWS steam
tables, until the oceans are gone). So heating the planet past ~100 C means boiling the oceans, and
the model pays the latent heat for every metre.

OLR(T) rises from today's 240 W/m^2 but saturates at the runaway-greenhouse limit (~282 W/m^2,
Goldblatt et al. 2013, Nature Geosci. 6:661) while there is liquid water: a wet planet cannot radiate
more than that, whatever its surface temperature. Once the ocean is all steam, OLR rises again above
~1,500 K (surface emission through the near-IR windows).

P_rain comes from the tracer stage (every parcel's 1/2 m v^2 at the top of the atmosphere), plus any
N-body fragment that hits the Earth. A fraction f_space of the entry energy is radiated straight back
to space from the hot upper atmosphere and plumes; the rest heats the lower atmosphere and surface.
"""
import math
import numpy as np
from numba import njit

from constants import A_EARTH, M_OCEAN, SIGMA, YEAR, DAY

G0 = 9.81
T0 = 288.0
COL_OCEAN = M_OCEAN / A_EARTH          # 2.6e6 kg/m^2: the ocean as a global layer 2.6 km deep
C_AIR = 1.03e4 * 1004.0                 # J/m^2/K
MIX_DEPTH = 70.0                        # m, ocean mixed layer (over 71% of the globe)
M_MIX = 1025.0 * MIX_DEPTH * 0.71       # kg/m^2 global average
K_DEEP = 0.7                            # W/m^2/K, ocean heat uptake efficiency (today)
F_MIX_MAX = 2.0                         # W/m^2: most heat tides and wind can mix down through a hot,
                                        # stably stratified surface layer (~2 TW of mixing power, 20%
                                        # efficiency, thermal expansion 3e-4/K, 1 km)
C_DEEP = 4.0e3 * (COL_OCEAN - M_MIX)    # J/m^2/K
S_ABS = 239.7                           # W/m^2 absorbed sunlight today
OLR_LIM = 282.0                         # runaway greenhouse limit
LAMBDA = 1.25                           # W/m^2/K climate feedback near today's climate (~3 K per CO2 doubling)

# saturated water: T (C), h_f, h_g (kJ/kg)  (IAPWS-IF97 steam tables)
_ST = np.array([
    [0.01, 0.0, 2500.9], [20, 83.9, 2537.4], [40, 167.5, 2573.5], [60, 251.2, 2608.8], [80, 335.0, 2643.0],
    [100, 419.2, 2675.6], [120, 503.8, 2705.9], [140, 589.2, 2733.4], [160, 675.5, 2757.4],
    [180, 763.1, 2777.2], [200, 852.3, 2791.0], [220, 943.6, 2798.7], [240, 1037.6, 2802.2],
    [260, 1135.0, 2796.6], [280, 1236.7, 2779.9], [300, 1344.8, 2748.7], [320, 1462.2, 2700.6],
    [340, 1594.5, 2622.2], [360, 1761.5, 2481.6], [370, 1892.6, 2333.5], [373.946, 2084.3, 2084.3]])
TC, PC = 647.096, 22.064e6


def p_sat(T):
    """Wagner & Pruss (2002) saturation pressure, Pa."""
    T = np.minimum(np.asarray(T, float), TC)
    tau = 1 - T / TC
    a = (-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502)
    s = a[0] * tau + a[1] * tau ** 1.5 + a[2] * tau ** 3 + a[3] * tau ** 3.5 + a[4] * tau ** 4 + a[5] * tau ** 7.5
    return PC * np.exp(TC / T * s)


def enthalpy_table():
    """H(T) per m^2 relative to today (T0), on a fine grid from 200 K to 6000 K (no rock melting)."""
    T = np.linspace(200, 6000, 58001)
    Tc_ = T - 273.15
    hf = np.interp(Tc_, _ST[:, 0], _ST[:, 1]) * 1e3
    hg = np.interp(Tc_, _ST[:, 0], _ST[:, 2]) * 1e3
    hf0 = np.interp(T0 - 273.15, _ST[:, 0], _ST[:, 1]) * 1e3
    m_vap = np.where(T < TC, np.minimum(p_sat(T) / G0, COL_OCEAN), COL_OCEAN)
    m_vap0 = p_sat(T0) / G0
    # steam above the critical point: heat as vapour, cp ~ 2.5 kJ/kg/K
    hg_hot = np.where(T > TC, 2084.3e3 + 2.5e3 * (T - TC), hg)
    m_liq_mix = np.maximum(M_MIX - np.maximum(m_vap - m_vap0, 0), 0)
    H = (C_AIR * (T - T0) + m_liq_mix * (np.minimum(hf, 2084.3e3) - hf0)
         + (m_vap - m_vap0) * (hg_hot - hf0))
    # land/rock skin: ~1 m of rock over 29% of the globe
    H += 0.29 * 2700 * 1000.0 * 1.0 * (T - T0)
    H -= np.interp(T0, T, H)
    # h_g falls steeply just below the critical point; keep H(T) monotonic (a short plateau)
    H = np.maximum.accumulate(H)
    return T, H, m_vap - m_vap0


TT, HH, MV = enthalpy_table()


@njit(cache=True)
def olr(T, ocean_left):
    if T <= T0:
        return S_ABS + LAMBDA * (T - T0)
    o = OLR_LIM - (OLR_LIM - S_ABS) * math.exp(-LAMBDA * (T - T0) / (OLR_LIM - S_ABS))
    if ocean_left <= 0.0 and T > 1500.0:
        o += 0.3 * SIGMA * (T ** 4 - 1500.0 ** 4)
    return o


@njit(cache=True)
def run_ebm(t, P, f_space, TT, HH, MV, col_ocean):
    """t: times (s, increasing), P: rain power per m^2 (W/m^2, interval means). Returns T, T_deep, vapour."""
    n = t.shape[0]
    Ts = np.zeros(n)
    Td = np.zeros(n)
    vap = np.zeros(n)
    olr_o = np.zeros(n)
    H = 0.0
    T = T0
    tdeep = T0
    Ts[0] = T
    Td[0] = tdeep
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        # sub-step for stability
        nsub = max(1, int(dt / (5 * DAY)))
        h = dt / nsub
        for _ in range(nsub):
            v = np.interp(T, TT, MV)
            left = col_ocean - v
            fd = min(K_DEEP * (T - tdeep), F_MIX_MAX)
            q = (1 - f_space) * P[i - 1] + S_ABS - olr(T, left) - fd
            H += q * h
            tdeep += fd * h / C_DEEP
            T = np.interp(H, HH, TT)
        Ts[i] = T
        Td[i] = tdeep
        vap[i] = np.interp(T, TT, MV)
        olr_o[i] = olr(T, col_ocean - vap[i])
    return Ts, Td, vap, olr_o
