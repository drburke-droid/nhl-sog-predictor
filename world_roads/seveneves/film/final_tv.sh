#!/bin/bash
# Final 1080p render of the waiting-room cut, here, resumable. Shots go in film order and only the frames the
# cut shows are rendered (film/tv.py used). After each shot: encode it to media/shots/<shot>_final.mp4 with its
# frame list, commit and push, so finished work survives a recycled container. Then assemble the film.
#   film/final_tv.sh            (re-run after any interruption: frames on disk and committed shots are kept)
set -u
B=${BLENDER:-/opt/blender_dl/blender-4.5.16-linux-x64/blender}
ROOT=/opt/seveneves_render
cd "$(dirname "$0")/.."
mkdir -p media/shots out/tv
python3 film/tv.py used out/tv/used_frames.json
BRANCH=$(git rev-parse --abbrev-ref HEAD)
commit() {
  git add "$@" && git commit -q -m "$MSG" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01S7EUN3W6exPjy1TpJCVruD" \
    && for i in 1 2 3 4; do git push -q -u origin "$BRANCH" && break || sleep $((2**i)); done
}
# the line-up of rocks is rendered at full size already
if [ ! -d $ROOT/final/P_lineup ] && [ -d $ROOT/animatic/P_lineup ]; then cp -r $ROOT/animatic/P_lineup $ROOT/final/; fi
SHOTS="H_ladder B_breakup E_rain S1_1m I1_1m I2_60m S2_60m S3_1km I3_1km J_ejecta E3_1km N3_1km S4_10km I4_10km E4_10km N4_10km S5_100km I5_100km K_entry E5_100km N5_100km F_steam"
for S in $SHOTS; do
  OUT=media/shots/${S}_final.mp4
  [ -f "$OUT" ] && { echo "skip $S"; continue; }
  "$B" -b -P film/render.py -- $S 0 9999 final $ROOT 1 out/tv/used_frames.json 2>&1 | grep --line-buffered -E "frame [0-9]+/|Error|Traceback"
  # all used frames present?
  need=$(python3 -c "import json;print(len(json.load(open('out/tv/used_frames.json'))['$S']))")
  have=$(ls $ROOT/final/$S/*.png 2>/dev/null | wc -l)
  [ "$have" -lt "$need" ] && { echo "INCOMPLETE $S $have/$need"; exit 1; }
  ls $ROOT/final/$S/*.png | xargs -n1 basename | sed 's/.png//' > media/shots/${S}_final_frames.txt
  ffmpeg -y -loglevel error -framerate 24 -pattern_type glob -i "$ROOT/final/$S/*.png" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow "$OUT" \
    && cp $ROOT/final/$S/times.json media/shots/${S}_final_times.json \
    && { [ -f $ROOT/final/$S/labels.json ] && cp $ROOT/final/$S/labels.json media/shots/${S}_final_labels.json; true; } \
    && MSG="Waiting-room film: final 1080p render of shot $S" commit "$OUT" media/shots/${S}_final_*
  echo "SHOT DONE $S"
done
# assemble: a master, and a copy under GitHub's 100 MB limit for the repository
python3 film/tv.py final out/tv/waiting_room_FINAL_master.mp4
cd out/tv
ffmpeg -y -loglevel error -i waiting_room_FINAL_master.mp4 -c:v libx264 -b:v 1700k -preset slow -pass 1 -an -f null /dev/null
ffmpeg -y -loglevel error -i waiting_room_FINAL_master.mp4 -c:v libx264 -b:v 1700k -preset slow -pass 2 -c:a aac -b:a 160k -movflags +faststart ../../media/waiting_room_FINAL.mp4
rm -f ffmpeg2pass*
cd ../..
MSG="Waiting-room film: final 1080p cut" commit media/waiting_room_FINAL.mp4 film/tv_script.md
echo "ALL DONE final"
