#!/bin/sh
# once every render is done: composite each shot, then the film
until [ -f /home/user/lot/render/chain.log ]; do sleep 60; done
cd /home/user/lot/src
for n in wright spitfire x1 x15; do
  rm -rf /home/user/lot/out/frames_$n
  python3 make_fly.py $n > /home/user/lot/render/comp_$n.log 2>&1 || echo FAIL $n >> /home/user/lot/render/post.log
done
python3 make_film.py > /home/user/lot/render/film.log 2>&1 || echo FAIL film >> /home/user/lot/render/post.log
echo POST_DONE >> /home/user/lot/render/post.log
