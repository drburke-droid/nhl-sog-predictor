"""Pack analysis.json into the compact data.json the viewer page reads.

python viewer/build_data.py runs/main/analysis.json viewer/data.json
"""
import json
import sys


def pack_replay(r):
    fr = r["frames"]
    P, A, K, O, S, U, E = [], [], [], [], [], [], []
    for i, f in enumerate(fr):
        for xy in f["p"]:
            P.extend(xy)
        A.append(sum(1 << k for k, a in enumerate(f["a"]) if a))
        K.extend(f["k"])
        O.append(f["o"])
        S.extend(f["s"])
        U.append(f["su"])
        if "e" in f:
            E.append([i, f["e"]])
    return dict(n=len(fr), P=P, A=A, K=K, O=O, S=S, U=U, E=E, final=r["final"])


def main(src, dst):
    a = json.load(open(src))
    fin = a["final"]
    out = dict(
        final={k: v for k, v in fin.items() if k != "replays"},
        replays=[pack_replay(r) for r in fin["replays"]],
        coach=a["coach"],
        snapshots=a.get("snapshots", []),
        log=a["log"],
    )
    with open(dst, "w") as f:
        json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
