# AE/AF lock trick video: intro, slowed demo with step cards, comparison, simple explainer, outro.
import cv2, numpy as np, json, sys, subprocess, math, os
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter, gaussian_filter1d
S='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/scratchpad'
IMG='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/images/'
OUT=sys.argv[1]
_fa=[a for a in sys.argv if a.startswith('--frames=')]
PREVIEW=[int(x) for x in _fa[0].split('=')[1].split(',')] if _fa else []
OW,OH=1080,1920; ASP=OW/OH; FPS=30
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
K=[(0,'s',768,1024,2048),(8,'s',768,1024,2048),(52,'p',60,0,1500),(110,'p',60,-20,1400),
   (150,'v',0,0,900),(218,'v',0,0,860),(240,'v',0,-30,700),(252,'v',0,-90,600),(268,'v',0,-20,800),
   (282,'p',40,-40,1300),(304,'p',40,-40,1300),(330,'v',0,0,800),(346,'v',0,0,760),(434,'v',0,0,640)]
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
VH_=1500.0
def band(img,z,panx=0.0):
    h=VH_/z; cx=RW/2+panx; cy=VH_/2+20*(z-1)/0.1; sc=OH/h
    M=np.array([[sc,0,OW/2-sc*cx],[0,sc,OH/2-sc*cy]],np.float32)
    return cv2.warpAffine(img,M,(OW,OH),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
COMPBG=bgplate(cv2.resize(_R,(SW,SH)),0.28)
# ---------------------------------------------------------------- SFX log
SFX=[]   # (time_s, kind)

# ================================================================ HOOK
I_LEN=156
def intro(f,T0):
    z=1.0+0.06*(f/I_LEN)
    a=band(LW_,z).astype(np.float32); b=band(_R,z).astype(np.float32)
    u=eio((f-72)/26); bg=(a*(1-u)+b*u)*(0.52-0.06*u)
    img=grade(bg.astype(np.uint8))
    if f<10: img=(img*ease(f/10)).astype(np.uint8)
    aA=ease((f-6)/14)*(1-ease((f-66)/10)); dy=20*(1-ease((f-6)/14))
    img=kicker(img,'A NEAT LITTLE TRICK',OW/2,690+dy,aA)
    put(img,tpatch('Make your\nphone camera\ndepend on\nyour glasses.',F_TB,104,WHITE,lh=1.08),OW/2,740+dy,aA,'mt')
    aB=ease((f-80)/14)*(1-ease((f-146)/10))
    put(img,tpatch('Take them away…',F_TS,62,GREY),OW/2,650+16*(1-ease((f-80)/14)),aB,'mt')
    aC=ease((f-94)/14)*(1-ease((f-146)/10))
    put(img,tpatch('…and it sees\nlike you do\nwithout them.',F_TB,104,WHITE,lh=1.08),OW/2,760+18*(1-ease((f-94)/14)),aC,'mt')
    return img
# ================================================================ DEMO
STEPS=[('Point your phone\nat something\nfar away','6 feet or more'),
       ('Hold your glasses\nin front of\nthe camera','In this video: a −3.00 lens'),
       ('Press and hold\nthe screen','until AE/AF LOCK appears'),
       ('Now take the\nglasses away','The focus stays locked'),
       ('Your camera now\nsees like you do\nwithout glasses','Snap a photo for your friends')]
CARD_N=78
SEGS=[('card',0,0),('play',0,52,1.0),('card',52,1),('play',52,150,0.85),('play',150,218,1.4),('card',218,2),
      ('play',218,252,0.8),('hold',252,42),('play',252,282,1.0),('card',282,3),('play',282,346,0.75),
      ('card',346,4),('play',346,434,0.75),('hold',434,12)]
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
CY=860
def step_card(img,step,i,n):
    d=ease(i/10)*(1-ease((i-(n-10))/10))
    if d>0:
        bl=cv2.GaussianBlur(img,(0,0),1+12*d); img=(bl.astype(np.float32)*(1-0.6*d)).astype(np.uint8)
    title,sub=STEPS[step]
    pt=tpatch(title,F_TB,92,WHITE,lh=1.08); ps=tpatch(sub,F_B,40,GREY)
    hk=34; ht=psize(pt)[1]; hs=psize(ps)[1]
    tot=hk+30+ht+24+hs; y=CY-tot/2
    a=ease((i-5)/11)*(1-ease((i-(n-14))/9)); dy=24*(1-ease((i-5)/11))-10*ease((i-(n-14))/9)
    img=kicker(img,f'STEP {step+1} OF 5',OW/2,y+hk/2+dy,a,size=28)
    put(img,pt,OW/2,y+hk+30+dy,a,'mt')
    put(img,ps,OW/2,y+hk+30+ht+24+dy,ease((i-10)/11)*(1-ease((i-(n-14))/9)),'mt')
    return img
def rbox(img,c,w,h,alpha,col=YEL,th=5):
    if alpha<=0: return img
    ov=img.copy(); x0,y0=int(c[0]-w/2),int(c[1]-h/2)
    cv2.rectangle(ov,(x0,y0),(int(x0+w),int(y0+h)),col,th,cv2.LINE_AA); return blend(img,ov,alpha)
def banner_pos(M,t,s):
    vc=vf_at(t); vw=290; return to_out(M,(vc[0]-0.01*vw, vc[1]-(0.5-0.036)*vw*4/3))
def demo(f,T0):
    e=DMAP[f]; kind=e[0]; t=e[1]
    img,M,s=live(t)
    if kind=='card':
        step,i,n=e[2],e[3],e[4]
        if i==0: SFX.append((T0+f/FPS,'whoosh'))
        if i==n-14: SFX.append((T0+f/FPS,'whoosh_out'))
        img=step_card(img,step,i,n)
    elif kind=='hold' and t==252:
        i,n=e[2],e[3]
        if i==2: SFX.append((T0+f/FPS,'lock'))
        a=ease(i/8)*(1-ease((i-(n-10))/10))
        bc=banner_pos(M,t,s); pulse=1+0.05*math.sin(i*0.4)
        img=rbox(img,bc,96*s*pulse,34*s*pulse,a,th=7)
        yp=bc[1]-17*s*pulse-150
        ov=img.copy(); cv2.line(ov,(int(bc[0]),int(bc[1]-17*s*pulse)),(int(bc[0]),int(yp+30)),YEL,4,cv2.LINE_AA); img=blend(img,ov,a)
        img=pill(img,bc[0],yp,tpatch('FOCUS LOCKED',F_K,36,(20,20,20),track=3),YEL,a)
    elif kind=='play':
        if 222<=t<=242:
            vc=vf_at(t); p=to_out(M,(vc[0]-0.13*290, vc[1]+0.05*387)); k=((t-222)%9)/9
            ov=img.copy(); cv2.circle(ov,(int(p[0]),int(p[1])),int((14+36*k)*s/2),(255,255,255),4,cv2.LINE_AA); img=blend(img,ov,0.8*(1-k))
        if 356<=t<=363:
            if not any(k=='shutter' for _,k in SFX): SFX.append((T0+f/FPS,'shutter'))
            a=0.42*math.exp(-((t-357.5)/1.6)**2); img=cv2.addWeighted(np.full_like(img,255),a,img,1-a,0)
    return img
# ================================================================ COMPARISON (stacked)
C_LEN=222
PW,PH=980,700; PX=(OW-PW)//2; PYS=[300,1040]
CR0=300
SH_=cv2.resize(LW_[CR0:CR0+int(RW*PH/PW)],(PW,PH),interpolation=cv2.INTER_AREA).astype(np.float32)
BL_=cv2.resize(_R[CR0:CR0+int(RW*PH/PW)],(PW,PH),interpolation=cv2.INTER_AREA).astype(np.float32)
PMASK=rrect_mask(PW,PH,28).astype(np.float32)[...,None]/255
COMPBG=bgplate(cv2.resize(_R,(1536,2048)),0.28)
def compare(f,T0):
    out=COMPBG.copy()
    for i,p in enumerate((SH_,BL_)):
        u=ease((f-6-i*12)/22); y=int(PYS[i]+50*(1-u))
        sh=np.zeros((OH,OW),np.float32); sh[y+18:y+PH+18,PX:PX+PW]=1; sh=cv2.GaussianBlur(sh,(0,0),22)[...,None]
        out=out*(1-0.55*u*sh); reg=out[y:y+PH,PX:PX+PW]; a=PMASK*u; out[y:y+PH,PX:PX+PW]=reg*(1-a)+p*a
    img=np.clip(out,0,255).astype(np.uint8)
    a=ease((f-2)/14)
    img=kicker(img,'A −3.00 NEARSIGHTED PERSON',OW/2,170+10*(1-a),a,size=28)
    put(img,tpatch('with vs. without glasses',F_TB,64,WHITE),OW/2,206+10*(1-a),a,'mt')
    for i,(t,fill) in enumerate((('WITH GLASSES',(250,250,250)),('WITHOUT GLASSES',YEL))):
        a=ease((f-30-i*14)/12)
        img=pill(img,PX+30,PYS[i]+58,tpatch(t,F_K,32,(20,20,20),track=3),fill,a,anchor='l')
    if f==6: SFX.append((T0+f/FPS,'swipe'))
    return img
# ================================================================ WHY (vertical stage)
BGc=np.zeros((OH,OW,3),np.float32); g=(yy/OH)[...,None]
BGc[:]=np.array([24,19,16],np.float32)*(1-g)+np.array([38,31,27],np.float32)*g
BGc=(BGc*np.clip(1-0.35*(((xx-OW/2)/(OW/1.3))**2+((yy-OH/2)/(OH/1.6))**2),0,1)[...,None]).astype(np.uint8)
AXY=900; CAMX=150; XS={'33':400,'1m':610,'6m':810,'far':990}
def lens_poly(cx,cy,h=230,thin=12,thick=44):
    pts=[(cx-thin/2-(thick-thin)/2*t*t, cy+t*h/2) for t in np.linspace(-1,1,40)]
    pts+=[(cx+thin/2+(thick-thin)/2*t*t, cy+t*h/2) for t in np.linspace(1,-1,40)]
    return np.array(pts,np.int32)
def phone_icon(img,a):
    ov=img.copy(); x0,x1,y0,y1=80,134,AXY-120,AXY+180
    cv2.rectangle(ov,(x0,y0),(x1,y1),(74,68,62),-1,cv2.LINE_AA)
    m=rrect_mask(x1-x0,y1-y0,22); sub=ov[y0:y1,x0:x1]; sub[m==0]=img[y0:y1,x0:x1][m==0]
    cv2.rectangle(ov,(x1,AXY-30),(CAMX,AXY+30),(110,102,94),-1,cv2.LINE_AA)
    cv2.ellipse(ov,(CAMX,AXY),(8,25),0,-90,90,(230,210,180),-1,cv2.LINE_AA)
    img=blend(img,ov,a); put(img,tpatch('your phone',F_B,30,DIM),(x0+x1)//2,y1+16,a,'mt'); return img
def padlock(img,cx,cy,a,sz=1.0):
    if a<=0: return img
    ov=img.copy(); w,h=int(34*sz),int(28*sz)
    cv2.ellipse(ov,(int(cx),int(cy-h/2)),(int(11*sz),int(14*sz)),0,180,360,YEL,int(5*sz),cv2.LINE_AA)
    cv2.rectangle(ov,(int(cx-w/2),int(cy-h/2)),(int(cx+w/2),int(cy+h/2)),YEL,-1,cv2.LINE_AA)
    cv2.circle(ov,(int(cx),int(cy)),int(4*sz),(30,30,30),-1,cv2.LINE_AA); return blend(img,ov,a)
def caption(img,txt,f,st,en):
    a=ease((f-st)/12)*(1-ease((f-en)/10))
    put(img,tpatch(txt,F_TS,54,WHITE,maxw=900,lh=1.15),OW/2,1180+14*(1-ease((f-st)/12)),a,'mt'); return img
W_LEN=288
def why(f,T0):
    img=BGc.copy()
    a=ease((f-4)/12)
    img=kicker(img,'WHY IT WORKS',OW/2,420,a,size=28)
    put(img,tpatch('The lens moves\nthe camera’s focus',F_TB,76,WHITE,lh=1.1),OW/2,460+12*(1-a),a,'mt')
    img=phone_icon(img,ease((f-2)/12))
    ua=ease((f-8)/22)
    ov=img.copy(); cv2.line(ov,(CAMX+12,AXY),(int(CAMX+12+(1010-CAMX)*ua),AXY),(150,156,166),2,cv2.LINE_AA); img=blend(img,ov,1)
    for key,lab in (('33','33 cm'),('1m','1 m'),('6m','6 m'),('far','far')):
        x=XS[key]; a=ease((f-14-(x-CAMX)/60)/8)
        ov=img.copy(); cv2.line(ov,(x,AXY-8),(x,AXY+8),(150,156,166),2,cv2.LINE_AA); img=blend(img,ov,a)
        put(img,tpatch(lab,F_B,32,GREY),x,AXY+42,a,'mt')
    lin=ease((f-34)/18); lout=ease((f-196)/20); la=lin*(1-lout)
    if f==36: SFX.append((T0+f/FPS,'pop'))
    if la>0:
        ly=AXY-200*(1-lin)-220*lout
        ov=img.copy(); cv2.fillPoly(ov,[lens_poly(245,int(ly))],(150,120,90),cv2.LINE_AA)
        cv2.polylines(ov,[lens_poly(245,int(ly))],True,(250,225,180),2,cv2.LINE_AA); img=blend(img,ov,0.9*la)
        put(img,tpatch('−3.00 lens',F_K,34,YELr),245,ly-136,la,'mb')
    mx=XS['far']+(XS['33']-XS['far'])*eio((f-62)/44); ma=ease((f-20)/12)
    if f==62: SFX.append((T0+f/FPS,'slide'))
    if ma>0:
        ov=img.copy(); cv2.fillPoly(ov,[np.array([(CAMX+8,AXY-28),(CAMX+8,AXY+28),(int(mx),AXY)],np.int32)],YEL,cv2.LINE_AA); img=blend(img,ov,0.16*ma)
        ov=img.copy(); cv2.circle(ov,(int(mx),AXY),28,YEL,3,cv2.LINE_AA); cv2.circle(ov,(int(mx),AXY),12,YEL,-1,cv2.LINE_AA); img=blend(img,ov,ma)
        put(img,tpatch('FOCUS',F_K,30,YELr,track=4),mx,AXY-74,ma,'mm')
    lk=ease((f-130)/10)
    if f==130: SFX.append((T0+f/FPS,'lock'))
    img=padlock(img,mx+90,AXY-74,lk,1.1)
    img=caption(img,'A −3.00 lens pulls the camera’s focus in to 33 cm.',f,64,124)
    img=caption(img,'AE/AF LOCK keeps it there.',f,132,186)
    img=caption(img,'Lens off: now the camera is nearsighted, just like you.',f,204,W_LEN)
    return img
# ================================================================ OUTRO
O_LEN=150
def outro(f,T0):
    z=1.03+0.04*(f/O_LEN); img=grade((band(_R,z).astype(np.float32)*0.42).astype(np.uint8))
    a=ease((f-8)/14)
    img=kicker(img,'TRY IT',OW/2,700,a,size=28)
    put(img,tpatch('Show your\nfriends how you\nreally see.',F_TB,104,WHITE,lh=1.08),OW/2,750+16*(1-a),a,'mt')
    put(img,tpatch('Works for nearsighted (minus) prescriptions.\nIt can’t show astigmatism.',F_B,34,GREY,lh=1.3),OW/2,1180,ease((f-24)/14),'mt')
    if f>O_LEN-24: img=(img*(1-ease((f-(O_LEN-24))/24))).astype(np.uint8)
    return img
# ================================================================ assemble
XF=14
SECTIONS=[('intro',intro,I_LEN),('demo',demo,D_LEN),('compare',compare,C_LEN),('explain1',why,W_LEN),('outro',outro,O_LEN)]
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
