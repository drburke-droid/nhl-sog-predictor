import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFilter
import importlib.util, sys
S='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/scratchpad'
spec=importlib.util.spec_from_file_location('th',S+'/work/thumbs.py')
src=open(S+'/work/thumbs.py').read().split('# ---------------- A:')[0]   # helpers only
g={}; exec(src,g)
W,H=1080,1920; g['W'],g['H']=W,H
LW,R=g['LW'],g['R']
def vband(img,y0=0,h=1500):
    w=int(h*W/H); x0=(1500-w)//2; return cv2.resize(img[y0:y0+h,x0:x0+w],(W,H),interpolation=cv2.INTER_CUBIC)
a=vband(LW); b=vband(R)
cx,cy,r=540,1010,330
yy,xx=np.mgrid[0:H,0:W]; m=cv2.GaussianBlur((((xx-cx)**2+(yy-cy)**2)<r*r).astype(np.float32),(0,0),1.2)[...,None]
img=(a*m+b.astype(np.float32)*0.6*(1-m)).astype(np.uint8)
im=g['to_pil'](img)
ring=Image.new('RGBA',(W,H),(0,0,0,0)); dr=ImageDraw.Draw(ring)
dr.ellipse((cx-r-16,cy-r-16,cx+r+16,cy+r+16),outline=(0,0,0,140),width=18)
dr.ellipse((cx-r-7,cy-r-7,cx+r+7,cy+r+7),outline=(222,190,120,255),width=12)
dr.line((cx+r+10,cy-50,W,cy-140),fill=(222,190,120,255),width=12)
im=Image.alpha_composite(im,ring)
im=g['text'](im,(W//2,330),'WHAT',g['BLACK'],120)
im=g['text'](im,(W//2,500),'−3.00',g['BLACK'],230,fill=g['YEL'])
im=g['text'](im,(W//2,660),'LOOKS LIKE',g['BLACK'],110)
im=g['pill'](im,W//2,1440,'TRY THIS PHONE TRICK',(250,250,250),size=36)
im.convert('RGB').save(S+'/work/aeaf_lock_short_cover.png'); print('ok')
