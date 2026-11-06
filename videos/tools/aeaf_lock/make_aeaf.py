# AE/AF lock trick video: intro, slowed demo with step cards, comparison, simple explainer, outro.
import cv2, numpy as np, json, sys, subprocess, math, os
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter, gaussian_filter1d
S='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/scratchpad'
IMG='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/images/'
OUT=sys.argv[1]
_fa=[a for a in sys.argv if a.startswith('--frames=')]
PREVIEW=[int(x) for x in _fa[0].split('=')[1].split(',')] if _fa else []
OW,OH=1920,1080; ASP=OW/OH; FPS=30
SW,SH=1536,2048
FD='/usr/share/fonts/opentype/inter/'
F_TB=FD+'InterDisplay-Bold.otf'; F_TS=FD+'InterDisplay-SemiBold.otf'; F_B=FD+'Inter-Medium.otf'; F_K=FD+'Inter-SemiBold.otf'; F_BB=FD+'Inter-Bold.otf'
YEL=(10,214,255); YELr=(255,214,10); WHITE=(255,255,255); GREY=(214,218,224); DIM=(150,156,166)
def ease(u): u=min(max(u,0.0),1.0); return u*u*u*(u*(u*6-15)+10)
def eio(u): u=min(max(u,0.0),1.0); return 0.5-0.5*math.cos(math.pi*u)
yy,xx=np.mgrid[0:OH,0:OW].astype(np.float32)
VIG=np.clip(1-0.32*(((xx-OW/2)/(OW/2))**2+((yy-OH/2)/(OH/2))**2)**1.3*0.5,0,1)[...,None]
# ---------------------------------------------------------------- text
_cache={}
def tpatch(txt,font,size,color=WHITE,maxw=None,lh=1.22,track=0,align='c'):
    key=(txt,font,size,color,maxw,lh,track,align)
    if key in _cache: return _cache[key]
    f=ImageFont.truetype(font,size); d=ImageDraw.Draw(Image.new('RGBA',(1,1)))
    def tl(s): return d.textlength(s,font=f)+track*max(0,len(s)-1)
    lines=[]
    for para in txt.split('\n'):
        if maxw is None: lines.append(para); continue
        cur=''
        for w in para.split():
            t=(cur+' '+w).strip()
            if tl(t)<=maxw or not cur: cur=t
            else: lines.append(cur); cur=w
        lines.append(cur)
    W=int(max(tl(l) for l in lines))+12; L=int(size*lh)
    im=Image.new('RGBA',(W,L*(len(lines)-1)+int(size*1.3)),(0,0,0,0)); dd=ImageDraw.Draw(im)
    for i,l in enumerate(lines):
        x=(W-tl(l))/2 if align=='c' else (W-tl(l)-6 if align=='r' else 6)
        if track==0: dd.text((x,i*L),l,font=f,fill=color+(255,))
        else:
            for ch in l: dd.text((x,i*L),ch,font=f,fill=color+(255,)); x+=d.textlength(ch,font=f)+track
    a=np.array(im).astype(np.float32); p=(a[...,2::-1].copy(),a[...,3:4]/255)
    _cache[key]=p; return p
def psize(p): return p[1].shape[1],p[1].shape[0]
def put(img,p,x,y,alpha=1.0,anchor='lt'):
    if alpha<=0.003: return img
    rgb,al=p; h,w=al.shape[:2]; x=int(x); y=int(y)
    if anchor[0]=='m': x-=w//2
    if anchor[0]=='r': x-=w
    if anchor[1]=='m': y-=h//2
    if anchor[1]=='b': y-=h
    x0,y0=max(0,x),max(0,y); x1,y1=min(OW,x+w),min(OH,y+h)
    if x1<=x0 or y1<=y0: return img
    a=al[y0-y:y1-y,x0-x:x1-x]*alpha
    reg=img[y0:y1,x0:x1].astype(np.float32)
    img[y0:y1,x0:x1]=(reg*(1-a)+rgb[y0-y:y1-y,x0-x:x1-x]*a).astype(np.uint8)
    return img
def blend(img,ov,a):
    if a<=0.003: return img
    return cv2.addWeighted(ov,a,img,1-a,0)
def pill(img,cx,cy,p,fill,alpha,padx=26,pady=14,anchor='m'):
    if alpha<=0: return img
    w,h=psize(p); W=w+2*padx; H=int(h*0.78)+2*pady
    x0=int(cx-W/2) if anchor=='m' else (int(cx) if anchor=='l' else int(cx-W)); y0=int(cy-H/2)
    ov=img.copy(); r=H//2
    cv2.rectangle(ov,(x0+r,y0),(x0+W-r,y0+H),fill,-1,cv2.LINE_AA)
    cv2.circle(ov,(x0+r,y0+r),r,fill,-1,cv2.LINE_AA); cv2.circle(ov,(x0+W-r,y0+r),r,fill,-1,cv2.LINE_AA)
    img=blend(img,ov,alpha)
    put(img,p,x0+W/2,y0+H/2+2,alpha,'mm'); return img
def kicker(img,txt,cx,cy,alpha,col=YELr,size=26,lines=True):
    p=tpatch(txt,F_K,size,col,track=5)
    put(img,p,cx,cy,alpha,'mm')
    if lines and alpha>0:
        w,_=psize(p); ov=img.copy(); c=(col[2],col[1],col[0])
        cv2.line(ov,(int(cx-w/2-90),int(cy)),(int(cx-w/2-26),int(cy)),c,2,cv2.LINE_AA)
        cv2.line(ov,(int(cx+w/2+26),int(cy)),(int(cx+w/2+90),int(cy)),c,2,cv2.LINE_AA)
        img=blend(img,ov,alpha)
    return img
def rrect_mask(w,h,r):
    m=np.zeros((h,w),np.uint8); cv2.rectangle(m,(r,0),(w-r,h),255,-1); cv2.rectangle(m,(0,r),(w,h-r),255,-1)
    for cx,cy in ((r,r),(w-r,r),(r,h-r),(w-r,h-r)): cv2.circle(m,(cx,cy),r,255,-1,cv2.LINE_AA)
    return m
def grade(img,vig=True):
    f=img.astype(np.float32)/255; g=f.mean(axis=2,keepdims=True)
    f=g+(f-g)*1.06; f=0.5+(f-0.5)*1.04
    if vig: f=f*VIG
    return np.clip(f*255,0,255).astype(np.uint8)
def sharpen(img,amt):
    if amt<=0: return img
    b=cv2.GaussianBlur(img,(0,0),1.4); return cv2.addWeighted(img,1+amt,b,-amt,0)
# ---------------------------------------------------------------- source footage + slow motion
_ra=[a for a in sys.argv if a.startswith('--range=')]
RANGE=tuple(int(x) for x in _ra[0].split('=')[1].split(',')) if _ra else None
cv2.setNumThreads(1 if RANGE else 0)
NS=int(cv2.VideoCapture(S+'/src/raw.mov').get(cv2.CAP_PROP_FRAME_COUNT))
SRC={}
def load_src(needed):
    cap=cv2.VideoCapture(S+'/src/raw.mov'); i=0; mx=max(needed) if needed else -1
    while i<=mx:
        ok,f=cap.read()
        if not ok: break
        if i in needed: SRC[i]=f
        i+=1
_dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
_gx,_gy=np.meshgrid(np.arange(SW,dtype=np.float32),np.arange(SH,dtype=np.float32))
_fc={}
def _flows(i):
    if i in _fc: return _fc[i]
    a=cv2.cvtColor(cv2.resize(SRC[i],None,fx=0.5,fy=0.5),cv2.COLOR_BGR2GRAY)
    b=cv2.cvtColor(cv2.resize(SRC[i+1],None,fx=0.5,fy=0.5),cv2.COLOR_BGR2GRAY)
    up=lambda f: cv2.resize(f,(SW,SH))*2
    r=(up(_dis.calc(a,b,None)),up(_dis.calc(b,a,None)))
    if len(_fc)>3: _fc.clear()
    _fc[i]=r; return r
def src_at(t):
    t=min(max(t,0),NS-1); i=int(math.floor(t)); a=t-i
    if a<0.02 or i>=NS-1: return SRC[min(i,NS-1)]
    if a>0.98: return SRC[i+1]
    f01,f10=_flows(i)
    w0=cv2.remap(SRC[i],_gx-a*f01[...,0],_gy-a*f01[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    w1=cv2.remap(SRC[i+1],_gx-(1-a)*f10[...,0],_gy-(1-a)*f10[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(w0,1-a,w1,a,0)
t_=json.load(open(S+'/work/track2.json'))
PC=np.array([[r['x']+r['s']*363/2, r['y']+r['s']*707/2] for r in t_])
PC=gaussian_filter1d(median_filter(PC,size=(11,1),mode='nearest'),6,axis=0,mode='nearest')
VF_OFF=np.array([-5.0,-38.0])
def pc_at(t):
    t=min(max(t,0),NS-1); i=int(t); a=t-i; j=min(i+1,NS-1)
    return PC[i]*(1-a)+PC[j]*a
def vf_at(t): return pc_at(t)+VF_OFF
# camera keyframes in SOURCE frames: (t, anchor, ox, oy, H)
K=[(0,'s',768,1024,2048),(8,'s',768,1024,2048),(52,'p',40,-40,900),(110,'p',40,-40,860),
   (150,'v',0,0,600),(218,'v',0,0,540),(240,'v',0,-30,450),(252,'v',0,-60,400),(268,'v',0,-20,500),
   (282,'p',30,-40,760),(304,'p',30,-40,760),(330,'v',0,0,450),(346,'v',0,0,420),(434,'v',0,0,350)]
def kc(k,t):
    a,ox,oy,H=k[1:]
    if a=='s': return np.array([ox,oy],float),H
    base=pc_at(t) if a=='p' else vf_at(t)
    return base+np.array([ox,oy]),H
def camera(t):
    for i in range(len(K)-1):
        if K[i][0]<=t<=K[i+1][0]: break
    a,b=K[i],K[i+1]; u=eio((t-a[0])/max(1e-6,b[0]-a[0]))
    ca,Ha=kc(a,t); cb,Hb=kc(b,t)
    H=math.exp(math.log(Ha)*(1-u)+math.log(Hb)*u); c=ca*(1-u)+cb*u; W=H*ASP
    c[0]=768 if W>=SW else min(max(c[0],W/2),SW-W/2); c[1]=min(max(c[1],H/2),SH-H/2)
    return c,H
SRCMASK=rrect_mask(SW,SH,48)
def bgplate(fr,dark=0.42):
    sm=cv2.GaussianBlur(cv2.resize(fr,(fr.shape[1]//8,fr.shape[0]//8),interpolation=cv2.INTER_AREA),(0,0),6)
    sc=max(OW/sm.shape[1],OH/sm.shape[0])*1.08
    big=cv2.resize(sm,None,fx=sc,fy=sc,interpolation=cv2.INTER_CUBIC)
    y0=(big.shape[0]-OH)//2; x0=(big.shape[1]-OW)//2
    return big[y0:y0+OH,x0:x0+OW].astype(np.float32)*dark
def live(t):
    fr=src_at(t); c,H=camera(t); s=OH/H
    M=np.array([[s,0,OW/2-s*c[0]],[0,s,OH/2-s*c[1]]],np.float32)
    fg=cv2.warpAffine(fr,M,(OW,OH),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
    fg=sharpen(fg,min(0.6,max(0,(s-1)*0.3)))
    if H*ASP<SW-1: out=fg
    else:
        m=cv2.warpAffine(SRCMASK,M,(OW,OH)).astype(np.float32)[...,None]/255
        sh=cv2.GaussianBlur(np.roll(m,14,axis=0),(0,0),22)[...,None]
        out=(fg*m+bgplate(fr)*(1-0.55*sh)*(1-m)).astype(np.uint8)
    return grade(out),M,s
def to_out(M,p): return M[:,:2]@np.array(p,np.float32)+M[:,2]
# ---------------------------------------------------------------- photos (aligned)
_L=cv2.imread(IMG+'1.jpg'); _R=cv2.imread(IMG+'2.jpg'); _M=np.load(S+'/work/align_M.npy')
RW,RH=_R.shape[1],_R.shape[0]
LW_=cv2.warpAffine(_L,_M,(RW,RH),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
WY0=230; WH_=int(RW/ASP)
def band(img,z,panx=0.0):
    h=WH_/z; cx=RW/2+panx; cy=WY0+WH_/2-30*(z-1)/0.1; sc=OH/h
    M=np.array([[sc,0,OW/2-sc*cx],[0,sc,OH/2-sc*cy]],np.float32)
    return cv2.warpAffine(img,M,(OW,OH),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
COMPBG=bgplate(cv2.resize(_R,(SW,SH)),0.28)
# ---------------------------------------------------------------- SFX log
SFX=[]   # (time_s, kind)
# ================================================================ INTRO
I_LEN=330
def intro(f,T0):
    z=1.0+0.09*(f/I_LEN); pan=-40+80*(f/I_LEN)
    a=band(LW_,z,pan).astype(np.float32); b=band(_R,z,pan).astype(np.float32)
    u=eio((f-176)/34)
    bg=a*(1-u)+b*u
    bg=bg*(0.50-0.06*u)
    img=grade(bg.astype(np.uint8))
    if f<22: img=(img*ease(f/22)).astype(np.uint8)
    # card A
    aA=ease((f-24)/18)*(1-ease((f-138)/14)); dy=22*(1-ease((f-24)/18))
    img=kicker(img,'A NEAT LITTLE TRICK',OW/2,388+dy,aA)
    put(img,tpatch('Make your phone camera\ndepend on your glasses.',F_TB,92,WHITE,lh=1.12),OW/2,440+dy,aA,'mt')
    # card B
    aB1=ease((f-156)/16)*(1-ease((f-308)/16)); aB2=ease((f-206)/18)*(1-ease((f-308)/16))
    put(img,tpatch('Then take the glasses away…',F_TS,58,GREY),OW/2,372+16*(1-ease((f-156)/16)),aB1,'mt')
    put(img,tpatch('…and it takes pictures just like\nyou see without them.',F_TB,84,WHITE,lh=1.12),OW/2,470+20*(1-ease((f-206)/18)),aB2,'mt')
    return img
# ================================================================ DEMO
STEPS=[('Point your phone at\nsomething far away','6 feet or more'),
       ('Hold your glasses in\nfront of the camera','In this video: a −3.00 lens'),
       ('Press and hold the screen','until AE/AF LOCK appears'),
       ('Now take the glasses away','The focus stays locked'),
       ('Your camera now sees like\nyou do without glasses','Snap a photo to show your friends')]
CARD_N=108
SEGS=[('card',0,0),('play',0,52,0.7),('card',52,1),('play',52,150,0.7),('play',150,218,1.0),('card',218,2),
      ('play',218,252,0.6),('hold',252,54),('play',252,282,0.8),('card',282,3),('play',282,346,0.6),
      ('card',346,4),('play',346,434,0.5),('hold',434,24)]
DMAP=[]
for sg in SEGS:
    if sg[0]=='card':
        for i in range(CARD_N): DMAP.append(('card',float(sg[1]),sg[2],i,CARD_N))
    elif sg[0]=='hold':
        for i in range(sg[2]): DMAP.append(('hold',float(sg[1]),i,sg[2]))
    else:
        _,a,b,sp=sg; n=int(round((b-a)/sp))
        for i in range(n): DMAP.append(('play',a+(b-a)*i/n))
D_LEN=len(DMAP)
def step_card(img,step,i,n):
    d=ease(i/14)*(1-ease((i-(n-14))/14))
    if d>0:
        bl=cv2.GaussianBlur(img,(0,0),1+11*d)
        img=(bl.astype(np.float32)*(1-0.58*d)).astype(np.uint8)
    title,sub=STEPS[step]
    pk=tpatch(f'STEP {step+1} OF 5',F_K,26,YELr,track=5)
    pt=tpatch(title,F_TB,84,WHITE,maxw=1350,lh=1.1); ps=tpatch(sub,F_B,38,GREY)
    hk=psize(pk)[1]; ht=psize(pt)[1]; hs=psize(ps)[1]
    tot=hk+30+ht+22+hs; y=OH/2-tot/2-10
    a=ease((i-8)/14)*(1-ease((i-(n-20))/12)); dy=26*(1-ease((i-8)/14))-10*ease((i-(n-20))/12)
    img=kicker(img,f'STEP {step+1} OF 5',OW/2,y+hk/2+dy,a)
    put(img,pt,OW/2,y+hk+30+dy,a,'mt')
    put(img,ps,OW/2,y+hk+30+ht+22+dy,ease((i-14)/14)*(1-ease((i-(n-20))/12)),'mt')
    return img
def rbox(img,c,w,h,alpha,col=YEL,th=5):
    if alpha<=0: return img
    ov=img.copy(); x0,y0=int(c[0]-w/2),int(c[1]-h/2)
    cv2.rectangle(ov,(x0,y0),(int(x0+w),int(y0+h)),col,th,cv2.LINE_AA)
    return blend(img,ov,alpha)
def banner_pos(M,t,s):
    vc=vf_at(t); vw=290
    return to_out(M,(vc[0]-0.01*vw, vc[1]-(0.5-0.036)*vw*4/3))
def demo(f,T0):
    e=DMAP[f]; kind=e[0]; t=e[1]
    img,M,s=live(t)
    if kind=='card':
        step,i,n=e[2],e[3],e[4]
        if i==0: SFX.append((T0+f/FPS,'whoosh'))
        if i==n-16: SFX.append((T0+f/FPS,'whoosh_out'))
        img=step_card(img,step,i,n)
    elif kind=='hold' and t==252:
        i,n=e[2],e[3]
        if i==2: SFX.append((T0+f/FPS,'lock'))
        a=ease(i/8)*(1-ease((i-(n-10))/10))
        bc=banner_pos(M,t,s); pulse=1+0.05*math.sin(i*0.4)
        img=rbox(img,bc,96*s*pulse,34*s*pulse,a,th=6)
        x1=bc[0]+48*s+10; xp=OW-300
        ov=img.copy(); cv2.line(ov,(int(x1),int(bc[1])),(int(xp-150),int(bc[1])),YEL,3,cv2.LINE_AA); img=blend(img,ov,a)
        img=pill(img,xp,bc[1],tpatch('FOCUS LOCKED',F_K,30,(20,20,20),track=3),YEL,a)
    elif kind=='play':
        if 222<=t<=242:
            vc=vf_at(t); p=to_out(M,(vc[0]-0.13*290, vc[1]+0.05*387)); k=((t-222)%9)/9
            ov=img.copy(); cv2.circle(ov,(int(p[0]),int(p[1])),int((14+36*k)*s/2),(255,255,255),4,cv2.LINE_AA); img=blend(img,ov,0.8*(1-k))
        if 356<=t<=363:
            if not any(k=='shutter' for _,k in SFX): SFX.append((T0+f/FPS,'shutter'))
            a=0.42*math.exp(-((t-357.5)/1.6)**2)
            img=cv2.addWeighted(np.full_like(img,255),a,img,1-a,0)
    return img
# ================================================================ COMPARISON
C1_LEN=228; C2_LEN=246; CXF=18; C_LEN=C1_LEN+C2_LEN-CXF
SHARP=LW_[0:1600]; BLUR=_R[0:1600]
PH=760; PW=int(PH*SHARP.shape[1]/SHARP.shape[0]); GAP=64
PX=[(OW-2*PW-GAP)//2,(OW-2*PW-GAP)//2+PW+GAP]; PY=188
PMASK=rrect_mask(PW,PH,24).astype(np.float32)[...,None]/255
P_SH=cv2.resize(SHARP,(PW,PH),interpolation=cv2.INTER_AREA).astype(np.float32)
P_BL=cv2.resize(BLUR,(PW,PH),interpolation=cv2.INTER_AREA).astype(np.float32)
def sbs(b):
    out=COMPBG.copy()
    for i,p in enumerate((P_SH,P_BL)):
        u=ease((b-8-i*10)/24); x=PX[i]; y=int(PY+44*(1-u))
        sh=np.zeros((OH,OW),np.float32); sh[y+18:y+PH+18,x:x+PW]=1; sh=cv2.GaussianBlur(sh,(0,0),22)[...,None]
        out=out*(1-0.55*u*sh)
        reg=out[y:y+PH,x:x+PW]; a=PMASK*u; out[y:y+PH,x:x+PW]=reg*(1-a)+p*a
    img=np.clip(out,0,255).astype(np.uint8)
    a=ease((b-4)/16)
    put(img,tpatch('How a −3.00 nearsighted person sees',F_TB,58,WHITE),OW/2,70+14*(1-a),a,'mt')
    for i,(t,fill,col) in enumerate((('WITH GLASSES',(250,250,250),(20,20,20)),('WITHOUT GLASSES',YEL,(20,20,20)))):
        a=ease((b-36-i*12)/14)
        img=pill(img,PX[i]+PW/2,PY+PH+52+10*(1-a),tpatch(t,F_K,28,col,track=3),fill,a)
    return img
def wipe_pos(b):
    if b<34: return 1.0
    if b<100: return 1.0-0.94*eio((b-34)/66)
    return 0.06+0.44*eio((b-108)/44)
def draw_handle(img,x):
    x=int(round(x)); ov=img.copy(); cv2.line(ov,(x+3,0),(x+3,OH),(0,0,0),6,cv2.LINE_AA); img=blend(img,ov,0.25)
    cv2.line(img,(x,0),(x,OH),(255,255,255),4,cv2.LINE_AA); cy=OH//2
    cv2.circle(img,(x,cy),34,(255,255,255),-1,cv2.LINE_AA)
    for sg in (-1,1):
        pts=np.array([[x+sg*8,cy-11],[x+sg*8,cy+11],[x+sg*21,cy]],np.int32); cv2.fillPoly(img,[pts],(30,30,30),cv2.LINE_AA)
    return img
def wipe(b):
    z=1.0+0.07*(b/C2_LEN)
    A=band(LW_,z); B=band(_R,z); x=wipe_pos(b)*OW
    m=np.clip((xx[...,None]-x)/2+0.5,0,1); img=(A*(1-m)+B*m).astype(np.uint8)
    if b>=34: img=draw_handle(img,x)
    a=ease((b-118)/14)
    img=pill(img,70,98,tpatch('WITH GLASSES',F_K,28,(20,20,20),track=3),(250,250,250),a,anchor='l')
    img=pill(img,OW-70,98,tpatch('WITHOUT GLASSES',F_K,28,(20,20,20),track=3),YEL,a,anchor='r')
    img=pill(img,OW/2,OH-80,tpatch('A −3.00 NEARSIGHTED PERSON',F_K,26,WHITE,track=4),(28,24,22),a*0.9)
    return img
def compare(f,T0):
    if f==C1_LEN-CXF+34: SFX.append((T0+f/FPS,'swipe'))
    if f<C1_LEN-CXF: return sbs(f)
    b=f-(C1_LEN-CXF)
    if b<CXF:
        u=ease(b/CXF); return cv2.addWeighted(wipe(b),u,sbs(f),1-u,0)
    return wipe(b)
# ================================================================ EXPLAINER
BGc=np.zeros((OH,OW,3),np.float32); g=(yy/OH)[...,None]
BGc[:]=np.array([24,19,16],np.float32)*(1-g)+np.array([38,31,27],np.float32)*g
BGc=(BGc*np.clip(1-0.35*(((xx-OW/2)/(OW/1.6))**2+((yy-OH/2)/(OH/1.3))**2),0,1)[...,None]).astype(np.uint8)
AXY=610; CAMX=262; XS={'33':640,'1m':1000,'6m':1370,'far':1730}
def lens_poly(cx,cy,h=260,thin=12,thick=48):
    pts=[(cx-thin/2-(thick-thin)/2*t*t, cy+t*h/2) for t in np.linspace(-1,1,40)]
    pts+=[(cx+thin/2+(thick-thin)/2*t*t, cy+t*h/2) for t in np.linspace(1,-1,40)]
    return np.array(pts,np.int32)
def phone_icon(img,a):
    ov=img.copy()
    x0,x1,y0,y1=184,244,AXY-130,AXY+190
    cv2.rectangle(ov,(x0,y0),(x1,y1),(74,68,62),-1,cv2.LINE_AA)
    m=rrect_mask(x1-x0,y1-y0,24); sub=ov[y0:y1,x0:x1]; sub[m==0]=img[y0:y1,x0:x1][m==0]
    cv2.rectangle(ov,(x1,AXY-34),(CAMX,AXY+34),(110,102,94),-1,cv2.LINE_AA)
    cv2.ellipse(ov,(CAMX,AXY),(9,28),0,-90,90,(230,210,180),-1,cv2.LINE_AA)
    img=blend(img,ov,a)
    put(img,tpatch('your phone',F_B,30,DIM),(x0+x1)//2,y1+18,a,'mt')
    return img
def padlock(img,cx,cy,a,sz=1.0):
    if a<=0: return img
    ov=img.copy(); w,h=int(34*sz),int(28*sz)
    cv2.ellipse(ov,(int(cx),int(cy-h/2)),(int(11*sz),int(14*sz)),0,180,360,YEL,int(5*sz),cv2.LINE_AA)
    cv2.rectangle(ov,(int(cx-w/2),int(cy-h/2)),(int(cx+w/2),int(cy+h/2)),YEL,-1,cv2.LINE_AA)
    cv2.circle(ov,(int(cx),int(cy)),int(4*sz),(30,30,30),-1,cv2.LINE_AA)
    return blend(img,ov,a)
def chart_card(defocus,w=300,h=216):
    big=Image.fromarray(np.full((h*2,w*2,3),245,np.uint8)); d=ImageDraw.Draw(big)
    d.text((w,h*0.62),'E',font=ImageFont.truetype(F_BB,150),fill=(15,15,15),anchor='mm')
    d.text((w,h*1.42),'F P',font=ImageFont.truetype(F_BB,74),fill=(15,15,15),anchor='mm')
    d.text((w,h*1.80),'T O Z',font=ImageFont.truetype(F_BB,42),fill=(15,15,15),anchor='mm')
    im=cv2.resize(np.array(big),(w,h),interpolation=cv2.INTER_AREA); r=defocus*5.5
    if r>0.6:
        k=int(2*math.ceil(r)+1); ker=np.zeros((k,k),np.float32); cv2.circle(ker,(k//2,k//2),int(round(r)),1,-1,cv2.LINE_AA); ker/=ker.sum()
        im=cv2.filter2D(im,-1,ker,borderType=cv2.BORDER_REPLICATE)
    return im
CARDS=[('33',chart_card(0),'clear',YELr),('1m',chart_card(2.0*1.2),'blurry',WHITE),('6m',chart_card(2.83*1.2),'very blurry',WHITE)]
def caption(img,txt,f,st,en):
    a=ease((f-st)/14)*(1-ease((f-en)/12))
    put(img,tpatch(txt,F_TS,46,WHITE,maxw=1500,lh=1.15),OW/2,905+14*(1-ease((f-st)/14)),a,'mt')
    return img
def title(img,k,t,f,st,en=None):
    a=ease((f-st)/14)*(1 if en is None else 1-ease((f-en)/12))
    img=kicker(img,k,OW/2,92,a)
    put(img,tpatch(t,F_TB,62,WHITE),OW/2,124+12*(1-ease((f-st)/14)),a,'mt'); return img
E1_LEN=640
def explain1(f,T0):
    img=BGc.copy()
    img=title(img,'WHY IT WORKS','The lens moves the camera’s focus',f,6,318)
    img=title(img,'WHAT YOU’LL SEE','Up close: clear. Far away: blurry.',f,334)
    img=phone_icon(img,ease((f-4)/14))
    ua=ease((f-12)/30)
    ov=img.copy(); cv2.line(ov,(CAMX+14,AXY),(int(CAMX+14+(1760-CAMX)*ua),AXY),(150,156,166),2,cv2.LINE_AA); img=blend(img,ov,1)
    for key,lab in (('33','33 cm'),('1m','1 m'),('6m','6 m'),('far','far away')):
        x=XS[key]; a=ease((f-20-(x-CAMX)/50)/10)
        ov=img.copy(); cv2.line(ov,(x,AXY-8),(x,AXY+8),(150,156,166),2,cv2.LINE_AA); img=blend(img,ov,a)
        put(img,tpatch(lab,F_B,32,GREY),x,AXY+42,a,'mt')
    # lens slides in (f 60-84), away (f 236-262)
    lin=ease((f-60)/24); lout=ease((f-236)/26); la=lin*(1-lout)
    if la>0:
        ly=AXY-200*(1-lin)-220*lout
        ov=img.copy(); cv2.fillPoly(ov,[lens_poly(360,int(ly))],(150,120,90),cv2.LINE_AA)
        cv2.polylines(ov,[lens_poly(360,int(ly))],True,(250,225,180),2,cv2.LINE_AA); img=blend(img,ov,0.9*la)
        put(img,tpatch('−3.00 lens',F_K,36,YELr),360,ly-150,la,'mb')
        if f==62: SFX.append((T0+f/FPS,'pop'))
    # focus marker: far -> 33 cm (f 92-150)
    mx=XS['far']+(XS['33']-XS['far'])*eio((f-92)/58)
    ma=ease((f-30)/14)
    if ma>0:
        ov=img.copy()
        cv2.fillPoly(ov,[np.array([(CAMX+8,AXY-30),(CAMX+8,AXY+30),(int(mx),AXY)],np.int32)],YEL,cv2.LINE_AA)
        img=blend(img,ov,0.16*ma)
        ov=img.copy(); cv2.circle(ov,(int(mx),AXY),30,YEL,3,cv2.LINE_AA); cv2.circle(ov,(int(mx),AXY),13,YEL,-1,cv2.LINE_AA); img=blend(img,ov,ma)
        fa=ma*(1-ease((f-330)/12))
        put(img,tpatch('FOCUS',F_K,30,YELr,track=4),mx,AXY-78,fa,'mm')
    if f==92: SFX.append((T0+f/FPS,'slide'))
    # lock (f 168)
    lk=ease((f-168)/10)*(1-ease((f-330)/12))
    if f==168: SFX.append((T0+f/FPS,'lock'))
    img=padlock(img,mx+92,AXY-74,lk,1.15)
    # cards (E2) at f 360+
    for j,(key,card,tag,tcol) in enumerate(CARDS):
        st=362+j*26; a=ease((f-st)/14)
        if f==st: SFX.append((T0+f/FPS,'pop'))
        if a<=0: continue
        h,w=card.shape[:2]; x=XS[key]; y0=AXY-70-h+int(16*(1-a))
        reg=img[y0:y0+h,x-w//2:x+w//2].astype(np.float32); img[y0:y0+h,x-w//2:x+w//2]=(reg*(1-a)+card*a).astype(np.uint8)
        put(img,tpatch(tag,F_TS,42,tcol),x,AXY+96,a,'mt')
    img=caption(img,'A −3.00 lens pulls the camera’s focus in close, to 33 cm.',f,96,160)
    img=caption(img,'Press and hold: AE/AF LOCK keeps it there.',f,174,232)
    img=caption(img,'Take the lens away. Now the camera is nearsighted, just like you.',f,250,318)
    img=caption(img,'Near things stay sharp. The farther away, the blurrier.',f,440,E1_LEN)
    return img
def star(sz=300,n=36):
    im=Image.new('L',(sz*2,sz*2),245); d=ImageDraw.Draw(im); c=sz
    for i in range(n):
        a0=2*math.pi*i/n; a1=2*math.pi*(i+0.5)/n
        d.polygon([(c,c),(c+sz*math.cos(a0),c+sz*math.sin(a0)),(c+sz*math.cos(a1),c+sz*math.sin(a1))],fill=20)
    m=Image.new('L',(sz*2,sz*2),0); ImageDraw.Draw(m).ellipse((4,4,sz*2-4,sz*2-4),fill=255)
    return cv2.resize(np.array(im),(sz,sz),interpolation=cv2.INTER_AREA),cv2.resize(np.array(m),(sz,sz),interpolation=cv2.INTER_AREA)
STAR,STARM=star(320)
def disc_blur(img,r):
    k=int(2*math.ceil(r)+1); ker=np.zeros((k,k),np.float32); cv2.circle(ker,(k//2,k//2),int(r),1,-1,cv2.LINE_AA); ker/=ker.sum()
    return cv2.filter2D(img,-1,ker,borderType=cv2.BORDER_REPLICATE)
def line_blur(img,L,ang):
    k=int(L)|1; ker=np.zeros((k,k),np.float32); c=k//2; dx_,dy_=math.cos(math.radians(ang))*c,math.sin(math.radians(ang))*c
    cv2.line(ker,(int(c-dx_),int(c-dy_)),(int(c+dx_),int(c+dy_)),1,2,cv2.LINE_AA); ker/=ker.sum()
    return cv2.filter2D(img,-1,ker,borderType=cv2.BORDER_REPLICATE)
ST_SPH=disc_blur(STAR,8); ST_CYL=line_blur(STAR,34,90)
def star_put(img,s,cx,cy,a):
    h,w=s.shape; x0,y0=int(cx-w//2),int(cy-h//2)
    rgb=np.repeat(s[...,None],3,2).astype(np.float32); m=(STARM.astype(np.float32)/255)[...,None]*a
    reg=img[y0:y0+h,x0:x0+w].astype(np.float32); img[y0:y0+h,x0:x0+w]=(reg*(1-m)+rgb*m).astype(np.uint8)
E3_LEN=300
def explain3(f,T0):
    img=BGc.copy()
    img=title(img,'ONE CATCH','It can’t show astigmatism',f,4)
    for j,(s,cx,t1,t2) in enumerate(((ST_SPH,620,'What the trick shows','even blur'),(ST_CYL,1300,'Astigmatism (cyl)','streaky blur'))):
        st=24+j*30; a=ease((f-st)/16)
        if f==st: SFX.append((T0+f/FPS,'pop'))
        if a<=0: continue
        star_put(img,s,cx,500+int(14*(1-a)),a)
        put(img,tpatch(t1,F_TS,42,WHITE),cx,700,a,'mt'); put(img,tpatch(t2,F_B,30,DIM),cx,754,a,'mt')
    img=caption(img,'A camera can only blur evenly, so the “cyl” in your Rx won’t show up.',f,96,E3_LEN)
    put(img,tpatch('Works for nearsighted (minus) prescriptions.',F_B,26,DIM),OW/2,1028,ease((f-150)/14),'mt')
    return img
# ================================================================ OUTRO
O_LEN=200
def outro(f,T0):
    z=1.04+0.05*(f/O_LEN)
    img=grade((band(_R,z).astype(np.float32)*0.42).astype(np.uint8))
    a=ease((f-14)/18)
    img=kicker(img,'TRY IT',OW/2,420,a)
    put(img,tpatch('Show your friends\nhow you really see.',F_TB,92,WHITE,lh=1.12),OW/2,470+18*(1-a),a,'mt')
    if f>O_LEN-36: img=(img*(1-ease((f-(O_LEN-36))/36))).astype(np.uint8)
    return img
# ================================================================ assemble
XF=16
SECTIONS=[('intro',intro,I_LEN),('demo',demo,D_LEN),('compare',compare,C_LEN),('explain1',explain1,E1_LEN),('explain3',explain3,E3_LEN),('outro',outro,O_LEN)]
STARTS=[]; t=0
for nm,fn,l in SECTIONS: STARTS.append(t); t+=l-XF
N=STARTS[-1]+SECTIONS[-1][2]
def frame(o):
    out=None
    for (nm,fn,l),s in zip(SECTIONS,STARTS):
        if s<=o<s+l:
            im=fn(o-s,s/FPS)
            if out is None: out=im
            else:
                u=ease((o-s)/XF); out=cv2.addWeighted(im,u,out,1-u,0)
    return out
meta=dict(N=N,fps=FPS,sections={nm:[s/FPS,(s+l)/FPS] for (nm,_,l),s in zip(SECTIONS,STARTS)})
def needed_src(frames):
    need=set()
    d0=STARTS[1]
    for o in frames:
        k=o-d0
        if 0<=k<D_LEN:
            t=DMAP[k][1]; i=int(math.floor(t)); need.update({i,min(i+1,NS-1)})
    return need
if PREVIEW: load_src(needed_src(PREVIEW))
if PREVIEW:
    for o in PREVIEW: cv2.imwrite(f'{OUT}_{o:05d}.jpg',frame(o),[cv2.IMWRITE_JPEG_QUALITY,88])
    print(json.dumps(meta)); sys.exit()
A,B=RANGE if RANGE else (0,N)
B=min(B,N); load_src(needed_src(range(A,B)))
p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{OW}x{OH}','-r',str(FPS),'-i','-',
    '-c:v','libx264','-preset','slow','-crf','17','-pix_fmt','yuv420p','-threads','2',OUT],stdin=subprocess.PIPE)
for o in range(A,B):
    p.stdin.write(frame(o).tobytes())
    if o%150==0: print(o,'/',N,flush=True)
p.stdin.close(); p.wait()
# de-duplicate sfx events
seen=set(); ev=[]
for t,k in SFX:
    key=(round(t,2),k)
    if key not in seen: seen.add(key); ev.append([t,k])
meta['sfx']=ev; json.dump(meta,open(OUT+'.json','w'),indent=1); print('done',A,B)
