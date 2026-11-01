#!/bin/sh
# The B-2 model and the imagery, none of which is kept in the repository.
set -e
mkdir -p /home/user/dl/fg && cd /home/user/dl/fg
# FlightGear's B-2 Spirit (Markus Zojer; FGAddon, GPL v2+), Aircraft-2024 set
[ -d B-2 ] || { curl -sSL -o B-2.zip https://mirrors.ibiblio.org/flightgear/ftp/Aircraft-2024/B-2.zip && unzip -q B-2.zip; }
cd "$(dirname "$0")"
# USDA NAIP over Whiteman AFB, 28 June 2022, 0.6 m (public domain), 3 km square
python fetch_naip.py whiteman -93.5479 38.7303 1500
# The side plate: BLM Utah's "Bonneville Salt Flats" (public domain), from Wikimedia Commons
mkdir -p /home/user/lot/b2/plates
curl -sS -A "B2FlybyResearch/1.0 (https://github.com/drburke-droid/nhl-sog-predictor)" \
  -o /home/user/lot/b2/plates/bonneville_blm_28776016261.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/1/14/Bonneville_Salt_Flats_%2828776016261%29.jpg"
