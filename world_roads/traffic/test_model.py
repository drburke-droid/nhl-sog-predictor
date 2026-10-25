"""Acceptance tests for the equilibrium model:  python -m pytest -q test_model.py"""
import math
import pytest
import model as M

T0, CAP = [12.0, 17.0, 22.0], [2000.0, 1500.0, 1100.0]


def test_zero_demand():
    ue, so = M.parallel_ue(T0, CAP, 0), M.parallel_so(T0, CAP, 0)
    for r in (ue, so):
        assert r["flows"] == [0.0, 0.0, 0.0]
        assert not any(r["used"])
        assert r["total_veh_min"] == 0 and r["max_gain_min"] == 0
        assert all(math.isfinite(t) for t in r["times"])
    assert math.isnan(ue["level"]) and math.isnan(M.price_of_anarchy(ue, so))


@pytest.mark.parametrize("D", [1, 500, 1999, 2600, 4000, 6000, 12000, 40000])
def test_ue_properties(D):
    r = M.parallel_ue(T0, CAP, D)
    assert all(f >= 0 for f in r["flows"])
    assert sum(r["flows"]) == pytest.approx(D, rel=1e-9)
    used = [t for t, u in zip(r["times"], r["used"]) if u]
    assert max(used) - min(used) < 1e-6                     # used routes share one time
    for t, u in zip(r["times"], r["used"]):
        if not u:
            assert t >= min(used) - 1e-9                    # unused routes are no faster
    assert r["max_gain_min"] < 1e-6


def test_symmetric_routes_split_equally():
    r = M.parallel_ue([15.0] * 3, [1200.0] * 3, 3000)
    assert r["flows"] == pytest.approx([1000.0] * 3, rel=1e-9)


@pytest.mark.parametrize("D", [300, 2500, 3500, 5000, 9000])
def test_so_no_worse_than_ue(D):
    ue, so = M.parallel_ue(T0, CAP, D), M.parallel_so(T0, CAP, D)
    assert so["total_veh_min"] <= ue["total_veh_min"] * (1 + 1e-12)
    assert M.price_of_anarchy(ue, so) >= 1 - 1e-12
    used = [m for m, u in zip(so["msc"], so["used"]) if u]
    assert max(used) - min(used) < 1e-6                     # SO equalizes marginal social costs
    assert sum(so["flows"]) == pytest.approx(D, rel=1e-9)


def test_joins_at_matches_equilibrium():
    j = M.joins_at(T0, CAP)
    assert j[0] == 0
    assert not M.parallel_ue(T0, CAP, j[1] * 0.99)["used"][1]
    assert M.parallel_ue(T0, CAP, j[1] * 1.01)["used"][1]
    assert not M.parallel_ue(T0, CAP, j[2] * 0.99)["used"][2]
    assert M.parallel_ue(T0, CAP, j[2] * 1.01)["used"][2]


def test_presets_are_honest():
    p = M.presets(T0, CAP)
    assert M.parallel_ue(T0, CAP, p["low"])["used"] == [True, False, False]
    assert M.parallel_ue(T0, CAP, p["moderate"])["used"] == [True, True, False]
    near = M.parallel_ue(T0, CAP, p["near"])
    assert near["used"] == [True, True, False] and T0[2] - near["level"] == pytest.approx(0.5, abs=0.01)
    assert all(M.parallel_ue(T0, CAP, p["heavy"])["used"])


def test_closed_and_bad_routes():
    r = M.parallel_ue([12.0, 17.0, 22.0], [2000.0, 0.0, -5.0], 3000)
    assert r["flows"][1] == 0 and r["flows"][2] == 0 and r["flows"][0] == pytest.approx(3000)
    assert r["open"] == [True, False, False]
    r = M.parallel_ue(T0, [0.0, 0.0, 0.0], 3000)              # nothing open: no flows, no NaN totals
    assert r["flows"] == [0.0] * 3 and r["total_veh_min"] == 0
    r = M.parallel_ue(T0, CAP, 3000, alpha=0.0)              # flat costs: all on the fastest route
    assert r["flows"] == pytest.approx([3000, 0, 0])


@pytest.mark.parametrize("alpha,beta,D", [(0.01, 1, 1e5), (1.0, 8, 1e5), (0.15, 4, 1e6), (2.0, 10, 1)])
def test_extremes_finite(alpha, beta, D):
    ue, so = M.parallel_ue(T0, CAP, D, alpha, beta), M.parallel_so(T0, CAP, D, alpha, beta)
    for r in (ue, so):
        assert all(math.isfinite(x) for x in r["flows"] + r["times"] + [r["total_veh_min"], r["avg_min"]])
        assert sum(r["flows"]) == pytest.approx(D, rel=1e-9)
    assert math.isfinite(M.price_of_anarchy(ue, so))


def test_braess():
    closed = M.network_ue(M.braess_network(connector=False))
    assert closed["path_times"] == pytest.approx([65.0, 65.0], abs=1e-6)
    assert closed["path_flows"] == pytest.approx([2000.0, 2000.0], abs=1e-3)
    opened = M.network_ue(M.braess_network(connector=True))
    i = opened["paths"].index("S-A-B-T")
    assert opened["path_flows"][i] == pytest.approx(4000.0, abs=1e-3)
    assert opened["path_times"][i] == pytest.approx(80.0, abs=1e-6)
    others = [t for k, t in enumerate(opened["path_times"]) if k != i]
    assert others == pytest.approx([85.0, 85.0], abs=1e-6)  # a lone switcher would take 85 min
    assert opened["total_veh_min"] > closed["total_veh_min"]


@pytest.mark.parametrize("D,paradox", [(2000, False), (3500, True), (6000, True), (9500, False)])
def test_braess_only_in_a_band(D, paradox):
    c = M.network_ue(M.braess_network(False, D))["avg_min"]
    o = M.network_ue(M.braess_network(True, D))["avg_min"]
    assert (o > c + 1e-6) == paradox
    assert o >= c - 1e-6 or D < 3000                          # below 3,000 the connector helps


@pytest.mark.parametrize("kw", [dict(replan=1.0), dict(replan=0.1, delay=3), dict(replan=0.5, sigma=2, noise=0.2, seed=9),
                                dict(replan=0.3, theta=2.0)])
def test_learning_conserves_vehicles(kw):
    r = M.learning(T0, CAP, 4200, days=80, **kw)
    for f in r["flows"]:
        assert all(x >= 0 for x in f)
        assert sum(f) == pytest.approx(4200, rel=1e-9)
    assert all(math.isfinite(g) for g in r["max_gain"])


def test_learning_is_repeatable_and_oscillation_shrinks():
    a = M.learning(T0, CAP, 4200, days=60, replan=0.5, sigma=1, noise=0.1, seed=3)
    b = M.learning(T0, CAP, 4200, days=60, replan=0.5, sigma=1, noise=0.1, seed=3)
    assert a["flows"] == b["flows"]
    # noise-free best response keeps a small cycle whose size shrinks with the share who re-plan
    amp = [max(M.learning(T0, CAP, 4200, days=400, replan=p)["avg_gain"][-30:]) for p in (0.01, 0.05, 0.2)]
    assert amp[0] < amp[1] < amp[2] and amp[0] < 0.2
    wild = M.learning(T0, CAP, 4200, days=200, replan=1.0)
    assert min(wild["avg_gain"][-20:]) > 1.0                    # everyone chasing yesterday's winner never settles
    calm = M.learning(T0, CAP, 4200, days=200, replan=0.2, theta=1.0)
    assert calm["flows"][-1] == pytest.approx(M.parallel_ue(T0, CAP, 4200)["flows"], abs=1.0)


def test_threshold_freezes_inside_a_band():
    r = M.learning(T0, CAP, 4200, days=400, replan=0.2, theta=2.0, start=[0, 1, 0])
    T, f = r["times"], r["flows"]
    assert f[-1] == f[-2]                                       # nobody moves any more ...
    gap = abs(T[-1][0] - T[-1][1])
    assert 0 < gap <= 2.0 + 1e-9                                # ... though the times are not equal


def test_app_selection():
    t = M.random_routes(20, 24.0, 1.0, seed=5)
    shown = M.app_shown(t, 3)
    assert len(shown) == 3 and t[shown[0]] == min(t)
    assert max(t[i] for i in shown) <= sorted(t)[2] + 1e-12
    assert M.app_shown([10.0, 30.0, 11.0], 3, max_detour=0.25) == [0, 2]


def test_mulberry_matches_reference():
    r = M.mulberry32(42)                                      # values from the JavaScript reference implementation
    assert [round(r(), 10) for _ in range(3)] == [0.6011037519, 0.4482905590, 0.8524657935]
