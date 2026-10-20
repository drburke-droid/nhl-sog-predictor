"""The contiguous US river network (NHDPlus V2, 1:100,000; EPA/USGS), one vector processing unit at a time.

For each unit: the flowlines, their value-added attributes (stream order, drainage area, path lengths), slopes,
and monthly precipitation and temperature for each flowline's catchment (PRISM 1971-2000, as shipped in
NHDPlus). Out of them, per unit:

  junctions   every confluence of two natural channels: the angle between the two tributaries, measured on
              straight lines from the junction to points 250, 500 and 1000 m up each channel; their orders and
              drainage areas; the aridity index (precipitation / potential evapotranspiration) there
  hack        every flowline: drainage area and the length of the longest channel upstream of its outlet
  horton      streams and their lengths by Strahler order

    python nhd.py            -> data/nhd/out/<vpu>.npz   (resumable)
"""
import os, re, sys, math, shutil, time, urllib.request, urllib.parse
import numpy as np
import py7zr
import shapefile
from dbfread import DBF

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKET = "https://dmap-data-commons-ow.s3.amazonaws.com/"
ROOT = "NHDPlusV21/Data/"
DATA = os.path.join(HERE, "data", "nhd")
OUT = os.path.join(DATA, "out")
REGIONS = ["NE", "MA", "SA", "GL", "MS", "SR", "TX", "RG", "CO", "GB", "PN", "CA"]     # the lower 48
DISTS = (250.0, 500.0, 1000.0)                                                          # m up each tributary
NATURAL = {46000, 46003, 46006, 46007}                                                   # StreamRiver fcodes


# ------------------------------------------------------------------ fetching
def keys(prefix):
    out, token = [], None
    while True:
        url = f"{BUCKET}?list-type=2&prefix={prefix}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        xml = urllib.request.urlopen(url).read().decode()
        out += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return out
        token = m.group(1)


def units():
    """{vpu: {kind: key}} for the lower 48"""
    import urllib.parse  # noqa: F401
    res = {}
    for reg in REGIONS:
        for k in keys(f"{ROOT}NHDPlus{reg}/"):
            m = re.search(r"NHDPlusV21_\w\w_(\w+?)_(NHDSnapshot|NHDPlusAttributes|VPUAttributeExtension)_\d+\.7z$", k)
            if m:
                res.setdefault(m.group(1), {})[m.group(2)] = k
    return {v: d for v, d in res.items() if len(d) == 3}


def get(key, dest):
    if os.path.exists(dest):
        return dest
    for attempt in range(6):
        try:
            urllib.request.urlretrieve(BUCKET + key, dest + ".part")
            os.replace(dest + ".part", dest)
            return dest
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError("download failed: " + key)


def extract(archive, wanted, into):
    """extract the members whose file names are in `wanted`; return {name: path}"""
    with py7zr.SevenZipFile(archive) as z:
        names = [n for n in z.getnames() if os.path.basename(n) in wanted]
    with py7zr.SevenZipFile(archive) as z:
        z.extract(path=into, targets=names)
    return {os.path.basename(n): os.path.join(into, n) for n in names}


# ------------------------------------------------------------------ reading
def dbf_cols(path, cols):
    t = DBF(path, load=False, ignore_missing_memofile=True)
    names = {f.lower(): f for f in t.field_names}
    cols_ = [names[c.lower()] for c in cols]
    out = {c: [] for c in cols}
    for r in t:
        for c, c_ in zip(cols, cols_):
            out[c].append(r[c_])
    return {c: np.array(v) for c, v in out.items()}


def climate_table(paths, prefix):
    """monthly values per FeatureID from NHDPlus's text tables"""
    ids, vals = None, []
    for m in range(1, 13):
        p = paths[f"{prefix}{m:02d}.txt"]
        a = np.genfromtxt(p, delimiter=",", skip_header=1, usecols=(0, 3), dtype=np.float64)
        order = np.argsort(a[:, 0])
        a = a[order]
        if ids is None:
            ids = a[:, 0].astype(np.int64)
        else:
            assert np.array_equal(ids, a[:, 0].astype(np.int64))
        vals.append(a[:, 1])
    return ids, np.stack(vals, 1)


# ------------------------------------------------------------------ climate
DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MIDDAY = np.cumsum(DAYS) - DAYS / 2


def daylength(lat):
    """mean daylength (hours) for each month at latitude lat (degrees); returns (n, 12)"""
    phi = np.radians(np.asarray(lat))[:, None]
    decl = 0.409 * np.sin(2 * np.pi * MIDDAY / 365.25 - 1.39)[None, :]
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1, 1))
    return 24 / np.pi * ws


def thornthwaite(T, lat):
    """monthly potential evapotranspiration (mm) from monthly mean temperature (C), Thornthwaite (1948);
    above 26.5 C, Willmott et al. (1985)"""
    Tp = np.maximum(T, 0)
    I = ((Tp / 5) ** 1.514).sum(1, keepdims=True)
    a = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239
    with np.errstate(divide="ignore", invalid="ignore"):
        pet = 16 * (10 * Tp / np.maximum(I, 1e-9)) ** a
    hot = T > 26.5
    pet = np.where(hot, -415.85 + 32.24 * T - 0.43 * T ** 2, pet)
    pet = np.where(T <= 0, 0, pet)
    return pet * daylength(lat) / 12 * DAYS[None, :] / 30


# ------------------------------------------------------------------ one unit
def process(vpu, ks):
    out = os.path.join(OUT, f"{vpu}.npz")
    if os.path.exists(out):
        return out
    work = os.path.join(DATA, "work_" + vpu)
    os.makedirs(work, exist_ok=True)
    arch = {k: get(v, os.path.join(work, os.path.basename(v))) for k, v in ks.items()}
    fl = extract(arch["NHDSnapshot"], {"NHDFlowline.shp", "NHDFlowline.dbf", "NHDFlowline.shx"}, work)
    at = extract(arch["NHDPlusAttributes"], {"PlusFlowlineVAA.dbf", "elevslope.dbf"}, work)
    want = {f"Incr{v}MM{m:02d}.txt" for v in ("Precip", "Temp") for m in range(1, 13)}
    cl = extract(arch["VPUAttributeExtension"], want, work)

    # flowlines: geometry and type
    sf = shapefile.Reader(fl["NHDFlowline.shp"], encoding="latin-1", encodingErrors="replace")
    fields = [f[0].upper() for f in sf.fields[1:]]
    i_com, i_fc = fields.index("COMID"), fields.index("FCODE")
    comid, fcode, geoms = [], [], []
    for sr in sf.iterShapeRecords():
        comid.append(int(sr.record[i_com]))
        fcode.append(int(sr.record[i_fc]) if sr.record[i_fc] is not None else 0)
        geoms.append(np.asarray(sr.shape.points, np.float64)[:, :2] if sr.shape.points else np.zeros((0, 2)))
    comid = np.array(comid, np.int64)
    fcode = np.array(fcode, np.int64)
    gi = {c: i for i, c in enumerate(comid)}

    vaa = dbf_cols(at["PlusFlowlineVAA.dbf"], ["ComID", "StreamOrde", "StreamCalc", "FromNode", "ToNode", "Hydroseq",
                                               "DnHydroseq", "Divergence", "StartFlag", "TerminalFl", "LengthKM",
                                               "TotDASqKM", "VPUIn", "Pathlength", "LevelPathI"])
    vc = vaa["ComID"].astype(np.int64)
    es = dbf_cols(at["elevslope.dbf"], ["COMID", "SLOPE"])
    slope = dict(zip(es["COMID"].astype(np.int64), es["SLOPE"].astype(np.float64)))

    # climate per catchment
    pid, P = climate_table(cl, "IncrPrecipMM")
    tid, T = climate_table(cl, "IncrTempMM")
    assert np.array_equal(pid, tid)
    P, T = P / 100.0, T / 100.0                              # mm, C
    # each catchment's latitude: its flowline's middle vertex
    lat_c = np.full(len(pid), np.nan)
    for j, c in enumerate(pid):
        g = geoms[gi[c]] if c in gi else None
        if g is not None and len(g):
            lat_c[j] = g[len(g) // 2, 1]
    ok = np.isfinite(lat_c)
    pet = np.full(len(pid), np.nan)
    pet[ok] = thornthwaite(T[ok], lat_c[ok]).sum(1)
    ai = dict(zip(pid, P.sum(1) / np.where(pet > 0, pet, np.nan)))
    pann = dict(zip(pid, P.sum(1)))

    # ---- junctions
    n = len(vc)
    to_node = vaa["ToNode"].astype(np.int64)
    from_node = vaa["FromNode"].astype(np.int64)
    div = vaa["Divergence"].astype(np.int64)
    order = vaa["StreamOrde"].astype(np.int64)
    calc = vaa["StreamCalc"].astype(np.int64)
    lenkm = vaa["LengthKM"].astype(np.float64)
    da = vaa["TotDASqKM"].astype(np.float64)
    vrow = {c: i for i, c in enumerate(vc)}
    incoming = {}
    for i in range(n):
        if div[i] == 2:
            continue                                   # minor divergences are not tributaries
        incoming.setdefault(to_node[i], []).append(i)
    outgoing = {}
    for i in range(n):
        outgoing.setdefault(from_node[i], []).append(i)
    rows = []
    for node, ins in incoming.items():
        if len(ins) != 2:
            continue
        outs = outgoing.get(node, [])
        if len(outs) != 1:
            continue
        trio = ins + outs
        if any(vc[i] not in gi or fcode[gi[vc[i]]] not in NATURAL for i in trio):
            continue
        if any(calc[i] != order[i] for i in trio):    # skip braided bits (where the calculated order differs)
            continue
        dirs, okk = [], True
        jx = jy = None
        for i in ins:
            g = geoms[gi[vc[i]]]
            if len(g) < 2:
                okk = False
                break
            end = g[-1]
            if jx is None:
                jx, jy = end
            elif abs(end[0] - jx) > 1e-6 or abs(end[1] - jy) > 1e-6:
                okk = False                            # the two tributaries must end at the same point
                break
            # walk up the tributary from the junction
            kx = math.cos(math.radians(jy)) * 111320.0
            ky = 110540.0
            xy = np.c_[(g[::-1, 0] - jx) * kx, (g[::-1, 1] - jy) * ky]
            seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
            cum = np.r_[0, np.cumsum(seg)]
            vs = []
            for d in DISTS:
                if cum[-1] < d:
                    vs.append((np.nan, np.nan))
                else:
                    k = np.searchsorted(cum, d)
                    t = (d - cum[k - 1]) / max(seg[k - 1], 1e-9)
                    p = xy[k - 1] + (xy[k] - xy[k - 1]) * t
                    vs.append((p[0], p[1]))
            dirs.append(vs)
        if not okk:
            continue
        angs = []
        for a_, b_ in zip(dirs[0], dirs[1]):
            if np.isnan(a_[0]) or np.isnan(b_[0]):
                angs.append(np.nan)
            else:
                c = (a_[0] * b_[0] + a_[1] * b_[1]) / (math.hypot(*a_) * math.hypot(*b_))
                angs.append(math.degrees(math.acos(max(-1, min(1, c)))))
        dn = outs[0]
        cdn = vc[dn]
        a1, a2 = (ins if da[ins[0]] >= da[ins[1]] else ins[::-1])
        rows.append([jx, jy, *angs, order[a1], order[a2], order[dn], da[a1], da[a2], da[dn],
                     ai.get(cdn, np.nan), pann.get(cdn, np.nan), slope.get(cdn, np.nan), lenkm[a1], lenkm[a2]])
    J = np.array(rows, np.float64)

    # ---- Hack: the longest channel upstream of each flowline's outlet
    hs = vaa["Hydroseq"].astype(np.int64)
    up_of = {}
    # minor divergences count too: NHDPlus's TotDASqKM is total upstream accumulation, so a reach below a
    # minor divergence drains everything above it, and its longest upstream channel runs through it
    for i in range(n):
        for j in outgoing.get(to_node[i], []):
            up_of.setdefault(j, []).append(i)
    lup = np.zeros(n)
    complete = np.ones(n, bool)
    vin = vaa["VPUIn"].astype(np.int64) != 0
    for i in np.argsort(-hs):                          # upstream first
        ups = up_of.get(i, [])
        lup[i] = lenkm[i] + (max(lup[u] for u in ups) if ups else 0.0)
        complete[i] = (not vin[i]) and all(complete[u] for u in ups)
    natural = np.array([fcode[gi[c]] in NATURAL if c in gi else False for c in vc])
    H = np.c_[da, lup, complete, natural, order].astype(np.float64)

    # ---- Horton: streams by Strahler order (a stream ends where the order rises or the network ends)
    dn_of = {}
    for i in range(n):
        o = outgoing.get(to_node[i], [])
        dn_of[i] = o[0] if len(o) == 1 else -1
    horton = []
    for w in range(1, 11):
        sel = np.nonzero((order == w) & (div != 2))[0]
        if not len(sel):
            continue
        ends = sum(1 for i in sel if dn_of[i] < 0 or order[dn_of[i]] != w)
        horton.append([w, ends, lenkm[sel].sum()])
    # ---- main stems: every level path (NHDPlus's main-stem chains), source to mouth
    lp = vaa["LevelPathI"].astype(np.int64)
    terminal = vaa["TerminalFl"].astype(np.int64)
    start = vaa["StartFlag"].astype(np.int64)        # 1 on headwater flowlines: a level path starting on one runs from a source, not a divergence
    paths = []
    order_lp = np.lexsort((-hs, lp))
    bounds = np.r_[0, np.nonzero(np.diff(lp[order_lp]))[0] + 1, len(order_lp)]
    R = 6371.0088
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order_lp[a:b]                            # upstream first
        top, bot = idx[0], idx[-1]
        if vc[top] not in gi or vc[bot] not in gi:
            continue
        g0, g1 = geoms[gi[vc[top]]], geoms[gi[vc[bot]]]
        if not len(g0) or not len(g1):
            continue
        (lo0, la0), (lo1, la1) = g0[0], g1[-1]
        p0, p1 = math.radians(la0), math.radians(la1)
        h = math.sin((p1 - p0) / 2) ** 2 + math.cos(p0) * math.cos(p1) * math.sin(math.radians(lo1 - lo0) / 2) ** 2
        straight = 2 * R * math.asin(min(1.0, math.sqrt(h)))
        nat = np.mean([fcode[gi[vc[i]]] in NATURAL if vc[i] in gi else False for i in idx])
        paths.append([lp[top], lenkm[idx].sum(), straight, da[bot], lo0, la0, lo1, la1, terminal[bot], complete[top] and complete[bot],
                      nat, slope.get(vc[bot], np.nan), ai.get(vc[bot], np.nan), len(idx), start[top]])
    np.savez_compressed(out, junctions=J, hack=H, horton=np.array(horton, np.float64), paths=np.array(paths, np.float64))
    shutil.rmtree(work, ignore_errors=True)
    return out


if __name__ == "__main__":
    import urllib.parse  # noqa: F401
    os.makedirs(OUT, exist_ok=True)
    us = units()
    print(len(us), "units:", " ".join(sorted(us)), flush=True)
    only = sys.argv[1:]
    for v in sorted(us):
        if only and v not in only:
            continue
        t0 = time.time()
        p = process(v, us[v])
        z = np.load(p)
        print(f"{v}: {len(z['junctions'])} junctions, {len(z['hack'])} flowlines, {time.time() - t0:.0f} s", flush=True)
