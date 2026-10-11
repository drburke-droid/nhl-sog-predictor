#!/usr/bin/env bash
# Full pipeline: data -> graph -> route -> layers -> preview + full film -> verification.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/ne
[ -f data/ne/ne_10m_land.geojson ] || curl -sSLo data/ne/ne_10m_land.geojson \
  https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson
NPROC=${NPROC:-8} python3 fetch_overture.py      # ~35 GB streamed, ~25 min; resumable
python3 build_graph.py
python3 route.py
python3 render.py prepare
python3 render.py preview
python3 render.py still 0 40 80 120 160 200 240 279.9
python3 render.py video --workers "${WORKERS:-3}"
python3 verify.py
