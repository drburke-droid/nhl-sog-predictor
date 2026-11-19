#!/bin/sh
# Final masters: the rendered film (python film.py video) plus the score (python music.py).
#   ./master.sh        1080p: two-pass H.264 at 5.2 Mb/s (under GitHub's 100 MB limit)
#                      -> out/every_shore_light_speed.mp4
#   RES=2 ./master.sh  4K: out/every_shore_light_speed_4K_master.mp4  the H.264 render (CRF 17) with the music, no
#                          re-encode: the file to upload (too big for git)
#                      out/every_shore_light_speed_4K.mp4  two-pass HEVC at 5.2 Mb/s, under 100 MB, for the repository
# Audio everywhere: AAC 256k under a -3 dBFS peak limiter (score.wav is already -14 LUFS).
set -e
cd "$(dirname "$0")/out"
RES=${RES:-1}
AF="volume=-1.2dB,alimiter=limit=0.708:attack=3:release=60:level=false"
V=coast_light_speed_r$RES.mp4
if [ "$RES" = "1" ]; then
  ffmpeg -y -v error -i $V -c:v libx264 -preset slow -b:v 5200k -pass 1 -passlogfile _p -tune film -pix_fmt yuv420p \
    -profile:v high -an -f mp4 /dev/null
  ffmpeg -y -v error -i $V -i score.wav -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 5200k -pass 2 -passlogfile _p \
    -tune film -pix_fmt yuv420p -profile:v high -c:a aac -b:a 256k -af "$AF" -shortest -movflags +faststart \
    every_shore_light_speed.mp4
  OUTS=every_shore_light_speed.mp4
else
  ffmpeg -y -v error -i $V -i score.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 256k -af "$AF" -shortest \
    -movflags +faststart every_shore_light_speed_4K_master.mp4
  X265="preset=slow:no-sao=1:aq-mode=3:psy-rd=1.5"
  ffmpeg -y -v error -i $V -c:v libx265 -preset slow -b:v 5200k -x265-params "$X265:pass=1:stats=_x265.log" \
    -pix_fmt yuv420p -an -f mp4 /dev/null
  ffmpeg -y -v error -i $V -i score.wav -map 0:v -map 1:a -c:v libx265 -preset slow -b:v 5200k \
    -x265-params "$X265:pass=2:stats=_x265.log" -tag:v hvc1 -pix_fmt yuv420p -c:a aac -b:a 256k -af "$AF" -shortest \
    -movflags +faststart every_shore_light_speed_4K.mp4
  OUTS="every_shore_light_speed_4K_master.mp4 every_shore_light_speed_4K.mp4"
fi
rm -f _p-0.log _p-0.log.mbtree _x265.log _x265.log.cutree
for f in $OUTS; do
  ls -la $f
  ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,duration -of compact $f
done
