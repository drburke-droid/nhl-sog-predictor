#!/bin/sh
# Final master: the rendered film (python film.py video) plus the score (python music.py), two-pass H.264 at
# 5.2 Mb/s so the file stays under GitHub's 100 MB limit, AAC 256k under a -3 dBFS peak limiter.
#   ./master.sh   -> out/every_shore_light_speed.mp4  (copied to ../media/coast/)
set -e
cd "$(dirname "$0")/out"
V=coast_light_speed_r1.mp4
ffmpeg -y -v error -i $V -c:v libx264 -preset slow -b:v 5200k -pass 1 -passlogfile _p -tune film -pix_fmt yuv420p \
  -profile:v high -an -f mp4 /dev/null
ffmpeg -y -v error -i $V -i score.wav -map 0:v -map 1:a -c:v libx264 -preset slow -b:v 5200k -pass 2 -passlogfile _p \
  -tune film -pix_fmt yuv420p -profile:v high -c:a aac -b:a 256k \
  -af "volume=-1.2dB,alimiter=limit=0.708:attack=3:release=60:level=false" -shortest -movflags +faststart \
  every_shore_light_speed.mp4
rm -f _p-0.log _p-0.log.mbtree
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,duration -of compact every_shore_light_speed.mp4
