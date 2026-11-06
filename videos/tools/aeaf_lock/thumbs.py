# YouTube thumbnails (1280x720) for the AE/AF lock video.
import cv2, numpy as np, math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
S='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/scratchpad'
IMG='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/images/'
FD='/usr/share/fonts/opentype/inter/'
BLACK=FD+'InterDisplay-Black.otf'; XB=FD+'InterDisplay-ExtraBold.otf'; SB=FD+'Inter-Bold.otf'
W,H=1280,720; YEL=(255,214,10)
L=cv2.imread(IMG+'1.jpg'); R=cv2.imread(IMG+'2.jpg'); M=np.load(S+'/work/align_M.npy')
LW=cv2.warpAffine(L,M,(R.shape[1],R.shape[0]),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
def band(img,y0=120,z=1.0):
    h=int(1500/(W/H)/z); w=int(1500/z); x0=(1500-w)//2
    c=img[y0:y0+h,x0:x0+w]; return cv2.resize(c,(W,H),interpolation=cv2.INTER_AREA)
def to_pil(bgr): return Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).convert('RGBA')
def grad(im,top=0.0,bottom=0.0,left=0.0,right=0.0):
    a=np.zeros((H,W),np.float32); y=np.linspace(0,1,H)[:,None]; x=np.linspace(0,1,W)[None,:]
    a=np.maximum(a,top*np.clip(1-y/0.55,0,1)**1.4); a=np.maximum(a,bottom*np.clip((y-0.6)/0.4,0,1)**1.4)
    a=np.maximum(a,left*np.clip(1-x/0.6,0,1)**1.2); a=np.maximum(a,right*np.clip((x-0.45)/0.55,0,1)**1.2)
    ov=Image.new('RGBA',(W,H),(0,0,0,0)); ov.putalpha(Image.fromarray((a*255).astype(np.uint8)))
    blk=Image.new('RGBA',(W,H),(8,8,12,255)); blk.putalpha(ov.getchannel('A')); return Image.alpha_composite(im,blk)
def text(im,xy,s,font,size,fill=(255,255,255),anchor='mm',stroke=7,shadow=True):
    f=ImageFont.truetype(font,size)
    if shadow:
        sh=Image.new('RGBA',im.size,(0,0,0,0)); ImageDraw.Draw(sh).text((xy[0]+4,xy[1]+8),s,font=f,fill=(0,0,0,170),anchor=anchor,stroke_width=stroke,stroke_fill=(0,0,0,170))
        im=Image.alpha_composite(im,sh.filter(ImageFilter.GaussianBlur(8)))
    d=ImageDraw.Draw(im); d.text(xy,s,font=f,fill=fill,anchor=anchor,stroke_width=stroke,stroke_fill=(12,12,16)); return im
def pill(im,cx,cy,s,fill,col=(18,18,18),size=30,anchor='m'):
    f=ImageFont.truetype(SB,size); d=ImageDraw.Draw(im); tw=d.textlength(s,font=f)+size*0.12*len(s)
    w=tw+48; h=size+30; x0=cx-w/2 if anchor=='m' else (cx if anchor=='l' else cx-w)
    d.rounded_rectangle((x0,cy-h/2,x0+w,cy+h/2),h/2,fill=fill)
    x=x0+24
    for ch in s: d.text((x,cy+1),ch,font=f,fill=col,anchor='lm'); x+=d.textlength(ch,font=f)+size*0.12
    return im
# ---------------- A: split skyline, MAKE YOUR PHONE NEARSIGHTED
a=band(LW,y0=60); b=band(R,y0=60); x=np.arange(W)[None,:,None]
img=np.where(x<W//2,a,b)
im=grad(to_pil(img),top=0.75,bottom=0.35)
d=ImageDraw.Draw(im); d.line((W//2,0,W//2,H),fill=(255,255,255,255),width=6)
im=text(im,(W//2,92),'MAKE YOUR PHONE',BLACK,92)
im=text(im,(W//2,200),'NEARSIGHTED',BLACK,128,fill=YEL)
im=pill(im,W//4,H-56,'WITH GLASSES',(250,250,250)); im=pill(im,3*W//4,H-56,'WITHOUT',YEL)
im.convert('RGB').save(S+'/work/thumb_A.png')
# ---------------- B: real phone, blurry screen in a sharp world
cap=cv2.VideoCapture(S+'/src/raw.mov'); cap.set(cv2.CAP_PROP_POS_FRAMES,400); ok,f=cap.read()
crop=f[640:640+864,0:1536]; img=cv2.resize(crop,(W,H),interpolation=cv2.INTER_AREA)
im=grad(to_pil(img),left=0.85)
im=text(im,(300,250),'MY EYES',BLACK,128,anchor='mm')
im=text(im,(300,385),'NO GLASSES',BLACK,92,fill=YEL,anchor='mm')
# arrow toward the phone screen
d=ImageDraw.Draw(im); p0=(470,470); p1=(640,420)
arr=Image.new('RGBA',im.size,(0,0,0,0)); da=ImageDraw.Draw(arr)
pts=[(p0[0]+(p1[0]-p0[0])*t, p0[1]+(p1[1]-p0[1])*t+40*math.sin(math.pi*t)) for t in np.linspace(0,1,30)]
da.line(pts,fill=(12,12,16,255),width=24,joint='curve'); da.line(pts,fill=YEL+(255,),width=12,joint='curve')
ang=math.atan2(pts[-1][1]-pts[-4][1],pts[-1][0]-pts[-4][0]); tip=(p1[0]+14*math.cos(ang),p1[1]+14*math.sin(ang))
head=[tip,(tip[0]-46*math.cos(ang-0.5),tip[1]-46*math.sin(ang-0.5)),(tip[0]-46*math.cos(ang+0.5),tip[1]-46*math.sin(ang+0.5))]
da.polygon(head,fill=YEL+(255,),outline=(12,12,16,255),width=5)
im=Image.alpha_composite(im,arr)
im=pill(im,300,520,'−3.00 PRESCRIPTION',(250,250,250),size=28)
im.convert('RGB').save(S+'/work/thumb_B.png')
# ---------------- C: lens circle shows the sharp view inside the blurry one
a=band(LW,y0=100,z=1.12); b=band(R,y0=100,z=1.12)
cx,cy,r=900,330,250
yy,xx=np.mgrid[0:H,0:W]; m=(((xx-cx)**2+(yy-cy)**2)<r*r).astype(np.float32)
m=cv2.GaussianBlur(m,(0,0),1.2)[...,None]
bb=(b.astype(np.float32)*0.62)
img=(a*m+bb*(1-m)).astype(np.uint8)
im=grad(to_pil(img),left=0.55)
ring=Image.new('RGBA',im.size,(0,0,0,0)); dr=ImageDraw.Draw(ring)
dr.ellipse((cx-r-14,cy-r-14,cx+r+14,cy+r+14),outline=(0,0,0,140),width=16)
dr.ellipse((cx-r-6,cy-r-6,cx+r+6,cy+r+6),outline=(222,190,120,255),width=10)
dr.ellipse((cx-r-1,cy-r-1,cx+r+1,cy+r+1),outline=(255,240,200,200),width=2)
dr.line((cx+r+8,cy-40,W,cy-110),fill=(222,190,120,255),width=10)   # temple arm
im=Image.alpha_composite(im,ring.filter(ImageFilter.GaussianBlur(0.6)))
im=text(im,(320,250),'WHAT',BLACK,96)
im=text(im,(320,380),'−3.00',BLACK,170,fill=YEL)
im=text(im,(320,505),'LOOKS LIKE',BLACK,82)
im.convert('RGB').save(S+'/work/thumb_C.png')
print('ok')
