import cv2, numpy as np, json, sys, subprocess, math
from PIL import Image, ImageDraw, ImageFont
S='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/scratchpad'
CAPTIONS = '--captions' in sys.argv
OUT = sys.argv[1]
_fa=[a for a in sys.argv if a.startswith('--frames=')]
PREVIEW=[int(x) for x in _fa[0].split('=')[1].split(',')] if _fa else []
OW,OH=1920,1080; ASP=OW/OH
SW,SH=1536,2048
FONT_B='/usr/share/fonts/opentype/inter/Inter-Bold.otf'
FONT_SB='/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
FONT_M='/usr/share/fonts/opentype/inter/Inter-Medium.otf'
YEL=(10,214,255)  # BGR of #FFD60A (iOS lock yellow)

# ---------- source frames ----------
cap=cv2.VideoCapture(S+'/src/raw.mov'); SRC=[]
while True:
    ok,f=cap.read()
    if not ok: break
    SRC.append(f)
NS=len(SRC)

# ---------- phone track (smoothed) ----------
t=json.load(open(S+'/work/track2.json'))
pc=np.array([[r['x']+r['s']*363/2, r['y']+r['s']*707/2] for r in t])
from scipy.ndimage import median_filter, gaussian_filter1d
pc=median_filter(pc,size=(11,1),mode='nearest')
pc=gaussian_filter1d(pc,6,axis=0,mode='nearest')
VF_OFF=np.array([-5,-38])
def vfc(sf): return pc[min(sf,NS-1)]+VF_OFF

# ---------- timeline ----------
FZ,FL=250,36          # freeze source frame 250 for 36 frames
NA=NS+FL              # live part length
NB=300                # comparison part
XF=16                 # crossfade live->comparison
N=NA+NB-XF
def src_of(o):
    if o<FZ: return o
    if o<FZ+FL: return FZ
    return min(o-FL,NS-1)

# camera keyframes: (out_frame, anchor, ox, oy, H)  anchor: 's' static abs center, 'p' phone, 'v' viewfinder
K=[(0,  's',768,1024,2048),
   (48, 's',790,1080,1880),
   (84, 'p',40,-40,860),
   (112,'p',40,-40,860),
   (142,'v',0,0,600),
   (214,'v',0,0,585),
   (240,'v',0,-20,470),
   (FZ, 'v',0,-20,470),
   (FZ+FL-4,'v',0,-95,380),
   (FZ+FL+16,'v',0,-10,600),
   (306,'v',0,0,600),
   (322,'p',40,-40,860),
   (346,'p',40,-40,860),
   (376,'v',0,0,560),
   (NA, 'v',0,0,505)]
def ease(u): u=min(max(u,0),1); return u*u*u*(u*(u*6-15)+10)
def kcenter(k,o):
    a,ox,oy,H=k[1:]
    sf=src_of(o)
    if a=='s': return np.array([ox,oy],float),H
    base=pc[sf] if a=='p' else vfc(sf)
    return base+np.array([ox,oy]),H
def camera(o):
    o=min(o,NA-1)
    for i in range(len(K)-1):
        if K[i][0]<=o<=K[i+1][0]: break
    a,b=K[i],K[i+1]
    u=ease((o-a[0])/max(1,b[0]-a[0]))
    ca,Ha=kcenter(a,o); cb,Hb=kcenter(b,o)
    H=Ha*(1-u)+Hb*u
    c=ca*(1-u)+cb*u
    W=H*ASP
    c[0]=768 if W>=SW else min(max(c[0],W/2),SW-W/2)
    c[1]=min(max(c[1],H/2),SH-H/2)
    return c,H
# ---------- helpers ----------
yy,xx=np.mgrid[0:OH,0:OW].astype(np.float32)
vig=1-0.28*(((xx-OW/2)/(OW/2))**2+((yy-OH/2)/(OH/2))**2)**1.3*0.5
vig=np.clip(vig,0,1)[...,None]
def rrect_mask(w,h,r):
    m=np.zeros((h,w),np.uint8); cv2.rectangle(m,(r,0),(w-r,h),255,-1); cv2.rectangle(m,(0,r),(w,h-r),255,-1)
    for cx,cy in ((r,r),(w-r,r),(r,h-r),(w-r,h-r)): cv2.circle(m,(cx,cy),r,255,-1,cv2.LINE_AA)
    return m
SRCMASK=rrect_mask(SW,SH,48)
def grade(img):
    f=img.astype(np.float32)/255
    g=f.mean(axis=2,keepdims=True)
    f=g+(f-g)*1.07                      # saturation
    f=0.5+(f-0.5)*1.05                  # contrast
    f=f*vig
    return np.clip(f*255,0,255).astype(np.uint8)
def bgplate(fr, dark=0.42):
    sm=cv2.resize(fr,(SW//8,SH//8),interpolation=cv2.INTER_AREA)
    sm=cv2.GaussianBlur(sm,(0,0),6)
    # cover-fit
    sc=max(OW/sm.shape[1],OH/sm.shape[0])*1.08
    big=cv2.resize(sm,None,fx=sc,fy=sc,interpolation=cv2.INTER_CUBIC)
    y0=(big.shape[0]-OH)//2; x0=(big.shape[1]-OW)//2
    return (big[y0:y0+OH,x0:x0+OW].astype(np.float32)*dark)
def sharpen(img,amt):
    if amt<=0: return img
    b=cv2.GaussianBlur(img,(0,0),1.4)
    return cv2.addWeighted(img,1+amt,b,-amt,0)
def live_frame(o):
    sf=src_of(o); fr=SRC[sf]
    c,H=camera(o); s=OH/H
    M=np.array([[s,0,OW/2-s*c[0]],[0,s,OH/2-s*c[1]]],np.float32)
    fg=cv2.warpAffine(fr,M,(OW,OH),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
    fg=sharpen(fg,min(0.6,max(0,(s-1)*0.35)))
    W=H*ASP
    if W<SW-1 and H<SH-1:
        out=fg
    else:
        m=cv2.warpAffine(SRCMASK,M,(OW,OH),flags=cv2.INTER_LINEAR).astype(np.float32)[...,None]/255
        sh=cv2.GaussianBlur(np.roll(m,14,axis=0),(0,0),22)[...,None]
        bg=bgplate(fr)*(1-0.55*sh)
        out=(fg*m+bg*(1-m)).astype(np.uint8)
    return grade(out),M,s
# ---------- comparison plates (user's real before/after photos, aligned) ----------
IMG='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/images/'
_L=cv2.imread(IMG+'1.jpg'); _R=cv2.imread(IMG+'2.jpg')
_M=np.load(S+'/work/align_M.npy')
RW,RH=_R.shape[1],_R.shape[0]
LW_=cv2.warpAffine(_L,_M,(RW,RH),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
SHARP=LW_[0:1600]; BLUR=_R[0:1600]
PH=820; PW=int(PH*SHARP.shape[1]/SHARP.shape[0])
GAP=70
PX=[(OW-2*PW-GAP)//2,(OW-2*PW-GAP)//2+PW+GAP]; PY=150 if CAPTIONS else (OH-PH)//2
PMASK=rrect_mask(PW,PH,26).astype(np.float32)[...,None]/255
COMPBG=bgplate(cv2.resize(_R,(SW,SH)),0.30)
# wipe region: 16:9 band over skyline + far tracks, where the two photos align
WY0=230; WH_=int(RW/ASP)
def wipe_src(img,z):
    h=WH_/z; w=RW/z; cx=RW/2; cy=WY0+WH_/2-30*(z-1)/0.1
    sc=OH/h
    M=np.array([[sc,0,OW/2-sc*cx],[0,sc,OH/2-sc*cy]],np.float32)
    return cv2.warpAffine(img,M,(OW,OH),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
def panel(img):
    return cv2.resize(img,(PW,PH),interpolation=cv2.INTER_AREA)
P_SHARP=panel(SHARP).astype(np.float32); P_BLUR=panel(BLUR).astype(np.float32)
# ---------- text overlays ----------
def text_layer(draw_fn):
    im=Image.new('RGBA',(OW,OH),(0,0,0,0)); d=ImageDraw.Draw(im); draw_fn(d,im)
    a=np.array(im).astype(np.float32); rgb=a[...,2::-1].copy(); al=a[...,3:4]/255
    return rgb,al
def wrap(d,text,font,maxw):
    words=text.split(); lines=[]; cur=''
    for w_ in words:
        tst=(cur+' '+w_).strip()
        if d.textlength(tst,font=font)<=maxw: cur=tst
        else: lines.append(cur); cur=w_
    lines.append(cur); return lines
def caption_card(num,title,sub=None):
    def fn(d,im):
        x0,y0=80,0; cw=470
        fN=ImageFont.truetype(FONT_B,30); fT=ImageFont.truetype(FONT_B,46); fS=ImageFont.truetype(FONT_M,30)
        tl=wrap(d,title,fT,cw-60); sl=wrap(d,sub,fS,cw-60) if sub else []
        h=40+44+18+len(tl)*58+(14+len(sl)*40 if sl else 0)+36
        y0=OH-110-h
        ov=Image.new('RGBA',(OW,OH),(0,0,0,0)); od=ImageDraw.Draw(ov)
        od.rounded_rectangle((x0,y0,x0+cw,y0+h),28,fill=(14,16,20,196))
        im.alpha_composite(ov)
        d.rounded_rectangle((x0+30,y0+36,x0+30+112,y0+36+44),22,fill=(255,214,10,255))
        d.text((x0+30+56,y0+36+22),f'STEP {num}',font=fN,fill=(18,18,18,255),anchor='mm')
        y=y0+36+44+22
        for l in tl: d.text((x0+30,y),l,font=fT,fill=(255,255,255,255)); y+=58
        if sl:
            y+=14
            for l in sl: d.text((x0+30,y),l,font=fS,fill=(205,210,218,255)); y+=40
    return text_layer(fn)
CAPS=[] 
if CAPTIONS:
    CAPS=[(10,62,caption_card(1,'Open your camera','Aim at something 6+ ft away')),
          (70,150,caption_card(2,'Hold your glasses over the lens','In this video: a −3.00 D lens')),
          (156,FZ+FL+12,caption_card(3,'Press and hold the screen','until AE/AF LOCK appears: focus is now locked')),
          (FZ+FL+18,362,caption_card(4,'Take the glasses away','The focus stays locked at your prescription')),
          (368,NA-6,caption_card(5,'This is your uncorrected vision','Snap a photo to show your friends'))]
def comp_labels():
    def fn(d,im):
        fT=ImageFont.truetype(FONT_B,44); fS=ImageFont.truetype(FONT_M,28)
        for i,(t1,t2) in enumerate((('With glasses','Your corrected vision'),('Without glasses','Your uncorrected vision, simulated'))):
            cx=PX[i]+PW//2
            d.text((cx,PY-78),t1,font=fT,fill=(255,255,255,255),anchor='mm')
            d.text((cx,PY-34),t2,font=fS,fill=(205,210,218,255),anchor='mm')
        fF=ImageFont.truetype(FONT_M,24)
        d.text((OW//2,PY+PH+52),'Works for nearsighted (minus) prescriptions, roughly -0.50 to -10.00 D',font=fF,fill=(170,176,186,255),anchor='mm')
    return text_layer(fn)
def wipe_labels():
    def fn(d,im):
        f=ImageFont.truetype(FONT_B,40)
        for x,t,anc in ((70,'WITH GLASSES','lm'),(OW-70,'WITHOUT GLASSES','rm')):
            w_=d.textlength(t,font=f); x0=x-18 if anc=='lm' else x-w_-18
            ov=Image.new('RGBA',(OW,OH),(0,0,0,0)); od=ImageDraw.Draw(ov)
            od.rounded_rectangle((x0,62,x0+w_+36,126),18,fill=(14,16,20,190)); im.alpha_composite(ov)
            d.text((x,94),t,font=f,fill=(255,255,255,255),anchor=anc)
    return text_layer(fn)
CL=comp_labels() if CAPTIONS else None
WL=wipe_labels() if CAPTIONS else None
NBP=118   # end of side-by-side phase
def wipe_pos(b):
    # 1.0 = all sharp; sweep to 0.06, then settle at 0.5
    if b<150: return 1.0
    if b<205: return 1.0-0.94*ease((b-150)/55)
    return 0.06+0.44*ease((b-212)/38)
def draw_handle(img,x):
    x=int(round(x))
    ov=img.copy()
    cv2.line(ov,(x+3,0),(x+3,OH),(0,0,0),6,cv2.LINE_AA)
    img=cv2.addWeighted(ov,0.25,img,0.75,0)
    cv2.line(img,(x,0),(x,OH),(255,255,255),4,cv2.LINE_AA)
    cy=OH//2
    cv2.circle(img,(x,cy),34,(255,255,255),-1,cv2.LINE_AA)
    for sgn in (-1,1):
        pts=np.array([[x+sgn*8,cy-11],[x+sgn*8,cy+11],[x+sgn*21,cy]],np.int32)
        cv2.fillPoly(img,[pts],(30,30,30),cv2.LINE_AA)
    return img
def sbs_frame(b):
    out=COMPBG.copy()
    for i,p in enumerate((P_SHARP,P_BLUR)):
        u=ease((b-6-i*8)/22)
        x=PX[i]; y=int(PY+40*(1-u))
        sh=np.zeros((OH,OW),np.float32); sh[y+16:y+PH+16,x:x+PW]=1; sh=cv2.GaussianBlur(sh,(0,0),20)[...,None]
        out=out*(1-0.5*u*sh)
        reg=out[y:y+PH,x:x+PW]; a=PMASK*u
        out[y:y+PH,x:x+PW]=reg*(1-a)+p*a
    out=np.clip(out,0,255).astype(np.uint8)
    if CL is not None:
        u=ease((b-20)/16); rgb,al=CL; a=al*u
        out=(out*(1-a)+rgb*a).astype(np.uint8)
    return out
def wipe_frame(b):
    z=1.0+0.08*ease((b-NBP)/(NB-NBP))
    A=wipe_src(LW_,z); Bm=wipe_src(_R,z)
    x=wipe_pos(b)*OW
    m=np.clip((xx[...,None]-x)/2+0.5,0,1)   # 0 left (sharp) .. 1 right (blur)
    out=(A*(1-m)+Bm*m).astype(np.uint8)
    if b>=150: out=draw_handle(out,x)
    if WL is not None and b>=222:
        u=ease((b-222)/14); rgb,al=WL; a=al*u
        out=(out*(1-a)+rgb*a).astype(np.uint8)
    return out
def comp_frame(b):
    if b<NBP: return sbs_frame(b)
    if b<NBP+18:
        u=ease((b-NBP)/18); return cv2.addWeighted(wipe_frame(b),u,sbs_frame(b),1-u,0)
    return wipe_frame(b)
# ---------- overlays on live part ----------
def ring(img,center,r,alpha,th=5,col=YEL):
    if alpha<=0: return img
    ov=img.copy(); cv2.circle(ov,(int(center[0]),int(center[1])),int(r),col,th,cv2.LINE_AA)
    return cv2.addWeighted(ov,alpha,img,1-alpha,0)
def rbox(img,c,w,h,alpha,col=YEL,th=5):
    if alpha<=0: return img
    ov=img.copy(); x0,y0=int(c[0]-w/2),int(c[1]-h/2)
    cv2.rectangle(ov,(x0,y0),(int(x0+w),int(y0+h)),col,th,cv2.LINE_AA)
    return cv2.addWeighted(ov,alpha,img,1-alpha,0)
def to_out(M,p): return M[:,:2]@np.array(p,np.float32)+M[:,2]
def live_overlays(img,o,M,s):
    sf=src_of(o)
    if CAPTIONS:
        # banner highlight (during freeze and a bit after)
        if FZ-4<=o<FZ+FL+10:
            a=ease((o-FZ+4)/8)*(1-ease((o-FZ-FL+2)/10))
            vc=vfc(sf); vw=290*pc_scale(sf)
            bc=to_out(M,(vc[0]-0.01*vw, vc[1]-(0.5-0.036)*vw*4/3))
            pulse=1+0.06*math.sin((o-FZ)*0.35)
            img=rbox(img,bc,95*s*pulse,34*s*pulse,a,th=6)
        # press indicator
        if 222<=o<FZ:
            vc=vfc(sf); vw=290*pc_scale(sf)
            pc_=to_out(M,(vc[0]-0.13*vw, vc[1]+0.05*vw*4/3))
            k=(o-222)%14/14
            img=ring(img,pc_,(12+40*k)*s/2.2,0.9*(1-k),th=5,col=(255,255,255))
        # shutter flash at src 358
        if 356<=sf<=362:
            a=[0.0,0.35,0.6,0.35,0.18,0.08,0.0][sf-356]
            img=cv2.addWeighted(np.full_like(img,255),a,img,1-a,0)
    return img
def pc_scale(sf): return 1.0
def apply_caps(img,o):
    for a,b,(rgb,al) in CAPS:
        if a<=o<b:
            u=ease((o-a)/9)*(1-ease((o-(b-9))/9))
            sh=int(-30*(1-ease((o-a)/9)))
            rr=np.roll(rgb,sh,axis=1); aa=np.roll(al,sh,axis=1)*u
            img=(img*(1-aa)+rr*aa).astype(np.uint8)
    return img
def frame(o):
    if o<NA-XF:
        img,M,s=live_frame(o); img=live_overlays(img,o,M,s); return apply_caps(img,o)
    b=o-(NA-XF)
    cf=comp_frame(b)
    if o<NA:
        img,M,s=live_frame(o); img=live_overlays(img,o,M,s); img=apply_caps(img,o)
        u=ease(b/XF); return cv2.addWeighted(cf,u,img,1-u,0)
    return cf
if PREVIEW:
    for o in PREVIEW:
        cv2.imwrite(f'{OUT}_{o:04d}.jpg',frame(o),[cv2.IMWRITE_JPEG_QUALITY,88])
    print('N',N,'NA',NA); sys.exit()
p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{OW}x{OH}','-r','30','-i','-',
    '-c:v','libx264','-preset','slow','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',OUT],stdin=subprocess.PIPE)
for o in range(N):
    p.stdin.write(frame(o).tobytes())
    if o%60==0: print(o,'/',N,flush=True)
p.stdin.close(); p.wait(); print('done',N)
