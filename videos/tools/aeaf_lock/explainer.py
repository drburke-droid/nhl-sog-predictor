import cv2, numpy as np, sys, subprocess, math
from PIL import Image, ImageDraw, ImageFont
OUT=sys.argv[1]
_fa=[a for a in sys.argv if a.startswith('--frames=')]
PREVIEW=[int(x) for x in _fa[0].split('=')[1].split(',')] if _fa else []
OW,OH=1920,1080
FB='/usr/share/fonts/opentype/inter/Inter-Bold.otf'; FSB='/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'; FM='/usr/share/fonts/opentype/inter/Inter-Medium.otf'
YEL=(10,214,255); YELr=(255,214,10); WHITE=(255,255,255); GREY=(200,206,214); DIM=(120,128,140)
BLUE=(250,200,120)  # BGR light blue for lens
def ease(u): u=min(max(u,0),1); return u*u*u*(u*(u*6-15)+10)
yy,xx=np.mgrid[0:OH,0:OW].astype(np.float32)
BG=np.zeros((OH,OW,3),np.float32)
g=(yy/OH)[...,None]
BG[:]=np.array([22,18,15],np.float32)*(1-g)+np.array([36,30,26],np.float32)*g
v=1-0.35*(((xx-OW/2)/(OW/1.6))**2+((yy-OH/2)/(OH/1.3))**2)
BG=(BG*v[...,None]).astype(np.uint8)
# ---------- text patches ----------
_cache={}
def tpatch(txt,font,size,color=WHITE,maxw=None,lh=1.3):
    key=(txt,font,size,color,maxw)
    if key in _cache: return _cache[key]
    f=ImageFont.truetype(font,size); d=ImageDraw.Draw(Image.new('RGBA',(1,1)))
    lines=[]
    for para in txt.split('\n'):
        if maxw is None: lines.append(para); continue
        cur=''
        for w in para.split():
            t=(cur+' '+w).strip()
            if d.textlength(t,font=f)<=maxw: cur=t
            else: lines.append(cur); cur=w
        lines.append(cur)
    W=int(max(d.textlength(l,font=f) for l in lines))+8; L=int(size*lh)
    im=Image.new('RGBA',(W,L*len(lines)+int(size*0.4)),(0,0,0,0)); dd=ImageDraw.Draw(im)
    for i,l in enumerate(lines): dd.text((2,i*L),l,font=f,fill=color+(255,))
    a=np.array(im).astype(np.float32); p=(a[...,2::-1].copy(),a[...,3:4]/255)
    _cache[key]=p; return p
def put(img,p,x,y,alpha=1.0,anchor='lt',dy=0):
    if alpha<=0: return
    rgb,al=p; h,w=al.shape[:2]
    if anchor[0]=='m': x-=w//2
    if anchor[0]=='r': x-=w
    if anchor[1]=='m': y-=h//2
    y+=int(dy)
    x0,y0=max(0,x),max(0,y); x1,y1=min(OW,x+w),min(OH,y+h)
    if x1<=x0 or y1<=y0: return
    a=al[y0-y:y1-y,x0-x:x1-x]*alpha
    reg=img[y0:y1,x0:x1].astype(np.float32)
    img[y0:y1,x0:x1]=(reg*(1-a)+rgb[y0-y:y1-y,x0-x:x1-x]*a).astype(np.uint8)
def fade_in(f,start,dur=14): return ease((f-start)/dur)
def txt_in(img,p,x,y,f,start,anchor='lt',out_at=None):
    a=fade_in(f,start)
    if out_at is not None: a*=1-ease((f-out_at)/10)
    put(img,p,x,y,a,anchor,dy=18*(1-fade_in(f,start)))
def header(img,f,kick,title,start=0):
    txt_in(img,tpatch(kick,FSB,26,YELr),110,86,f,start)
    txt_in(img,tpatch(title,FB,58),110,124,f,start+4)
def blend(img,ov,a):
    if a<=0: return img
    return cv2.addWeighted(ov,a,img,1-a,0)
def aline(img,p0,p1,col,th,a=1.0,dashed=False):
    if a<=0: return img
    ov=img.copy()
    if dashed:
        L=math.hypot(p1[0]-p0[0],p1[1]-p0[1]); n=max(1,int(L/22))
        for i in range(n):
            if i%2: continue
            u0=i/n; u1=min(1,(i+1)/n)
            q0=(int(p0[0]+(p1[0]-p0[0])*u0),int(p0[1]+(p1[1]-p0[1])*u0)); q1=(int(p0[0]+(p1[0]-p0[0])*u1),int(p0[1]+(p1[1]-p0[1])*u1))
            cv2.line(ov,q0,q1,col,th,cv2.LINE_AA)
    else: cv2.line(ov,(int(p0[0]),int(p0[1])),(int(p1[0]),int(p1[1])),col,th,cv2.LINE_AA)
    return blend(img,ov,a)
def arrowhead(img,p,direction,col,a=1.0,s=16):
    if a<=0: return img
    ov=img.copy(); dx,dy=direction; n=math.hypot(dx,dy); dx/=n; dy/=n
    pts=np.array([[p[0],p[1]],[p[0]-s*dx+0.55*s*dy,p[1]-s*dy-0.55*s*dx],[p[0]-s*dx-0.55*s*dy,p[1]-s*dy+0.55*s*dx]],np.int32)
    cv2.fillPoly(ov,[pts],col,cv2.LINE_AA); return blend(img,ov,a)
def partial(p0,p1,u): return (p0[0]+(p1[0]-p0[0])*u, p0[1]+(p1[1]-p0[1])*u)

# ======================= SCENE 1: how it works =======================
S1=0; S1L=420
AX=480; CAMX=360; LX=500; F=420; VX=LX+F   # virtual point at 33 cm
SENX=170; HS=[-88,-40,40,88]
RAY=(225,230,235)
def lens_poly(cx,cy,h=250,thin=10,thick=44):
    pts=[]
    for t in np.linspace(-1,1,40): pts.append((cx-thin/2-(thick-thin)/2*t*t, cy+t*h/2))
    for t in np.linspace(1,-1,40): pts.append((cx+thin/2+(thick-thin)/2*t*t, cy+t*h/2))
    return np.array(pts,np.int32)
def camera(img,a=1.0):
    ov=img.copy()
    cv2.rectangle(ov,(SENX-30,AX-150),(CAMX-12,AX+150),(62,56,50),-1,cv2.LINE_AA)
    cv2.rectangle(ov,(SENX-30,AX-150),(CAMX-12,AX+150),(120,112,104),2,cv2.LINE_AA)
    cv2.ellipse(ov,(CAMX-12,AX),(26,128),0,0,360,(200,180,150),-1,cv2.LINE_AA)
    cv2.ellipse(ov,(CAMX-12,AX),(26,128),0,0,360,(250,225,190),2,cv2.LINE_AA)
    cv2.line(ov,(SENX,AX-110),(SENX,AX+110),(150,150,150),6,cv2.LINE_AA)
    img=blend(img,ov,a)
    return img
def scene1(f):
    img=BG.copy()
    header(img,f,'HOW IT WORKS','The camera borrows your prescription')
    img=camera(img,fade_in(f,8))
    put(img,tpatch('Phone camera',FM,26,GREY),(SENX+CAMX)//2-15,AX+168,fade_in(f,10),'mt')
    put(img,tpatch('sensor',FM,22,DIM),SENX,AX-178,fade_in(f,10),'mb')
    removed=ease((f-262)/20)
    lens_a=fade_in(f,22)*(1-removed)
    if lens_a>0:
        ov=img.copy(); cv2.fillPoly(ov,[lens_poly(LX,AX)],(150,120,90),cv2.LINE_AA)
        cv2.polylines(ov,[lens_poly(LX,AX)],True,(250,220,170),2,cv2.LINE_AA)
        img=blend(img,ov,0.85*lens_a)
        put(img,tpatch('−3.00 D lens',FB,30,YELr),LX,AX-170,lens_a,'mb')
    # incoming parallel rays (right -> lens, or all the way to the camera once the lens is gone)
    u_in=ease((f-34)/40)
    for h in HS:
        y=AX+h; xe=LX+6-(LX-CAMX+6)*removed
        p0=(1820,y); p1=(xe,y)
        img=aline(img,p0,partial(p0,p1,u_in),RAY,3,0.95)
    if u_in>0.98:
        for h in HS: img=arrowhead(img,(1240,AX+h),(-1,0),RAY,1)
    txt_in(img,tpatch('Light from far away\narrives in parallel',FM,28,GREY,lh=1.25),1560,AX-190,f,40)
    # lens -> camera: diverging (with lens), straight (lens removed)
    u_out=ease((f-78)/24)
    ycam={}
    for h in HS:
        yd=AX+h*(1+(LX-CAMX)/F); ys=AX+h
        ye=yd*(1-removed)+ys*removed; ycam[h]=ye
        if removed<1:
            p0=(LX-6,AX+h); p1=(CAMX,ye)
            img=aline(img,p0,partial(p0,p1,u_out),RAY,3,0.95*(1-removed))
    # inside the camera: converge to a focus point
    u_c=ease((f-100)/18)
    xf_sharp=SENX; xf_far=SENX+70
    xf=xf_sharp*(1-removed)+xf_far*removed
    for h in HS:
        y0=ycam[h]
        p0=(CAMX-12,y0); pf=(xf,AX)
        # continue past focus to the sensor plane
        t=(CAMX-12-SENX)/max(1,(CAMX-12-xf)); pe=(SENX,y0+(AX-y0)*t)
        img=aline(img,p0,partial(p0,pe,u_c),YEL if removed>0.5 else RAY,2,0.95)
    sharp_a=ease((f-118)/12)*(1-removed)
    if sharp_a>0:
        ov=img.copy(); cv2.circle(ov,(SENX,AX),8,YEL,-1,cv2.LINE_AA); img=blend(img,ov,sharp_a)
        put(img,tpatch('sharp',FSB,24,YELr),SENX-40,AX,sharp_a,'rm')
    # dashed back-extensions to virtual point
    u_d=ease((f-125)/30)*(1-removed)
    if u_d>0:
        for h in HS:
            p0=(LX,AX+h); p1=(VX,AX)
            img=aline(img,p0,partial(p0,p1,ease((f-125)/30)),YEL,2,u_d,dashed=True)
        ov=img.copy(); cv2.circle(ov,(VX,AX),9,YEL,-1,cv2.LINE_AA); img=blend(img,ov,u_d)
    a=ease((f-140)/14)*(1-removed)
    if a>0:
        yb=AX+150
        img=aline(img,(LX,yb),(VX,yb),YEL,2,a); img=aline(img,(LX,yb-10),(LX,yb+10),YEL,2,a); img=aline(img,(VX,yb-10),(VX,yb+10),YEL,2,a)
        put(img,tpatch('seems to come from 33 cm',FSB,26,YELr),(LX+VX)//2,yb+16,a,'mt')
    # lens removed: focus plane stays at 33 cm, far light makes a blur spot
    lk=ease((f-285)/18)
    if lk>0:
        img=aline(img,(VX,AX-150),(VX,AX+150),YEL,3,lk,dashed=True)
        put(img,tpatch('Focus still locked at 33 cm',FSB,28,YELr),VX,AX-162,lk,'mb')
        ov=img.copy(); cv2.ellipse(ov,(SENX,AX),(9,int(88*(CAMX-12-SENX-70)/(CAMX-12-SENX-70+70)*0+60)),0,0,360,YEL,-1,cv2.LINE_AA); img=blend(img,ov,0.45*lk)
        put(img,tpatch('blur spot',FSB,24,YELr),SENX-40,AX,lk,'rm')
    lines=[(160,'1','Through a −3.00 D lens, light from far away spreads out as if it started 33 cm in front of the lens.'),
           (205,'2','Autofocus focuses at 33 cm, and AE/AF LOCK keeps it there.'),
           (300,'3','Take the lens away and the focus stays at 33 cm, just like an uncorrected −3.00 D eye.')]
    for i,(st,n,t) in enumerate(lines):
        y=780+i*82
        a=fade_in(f,st)
        if a>0:
            ov=img.copy(); cv2.circle(ov,(132,y+20),22,YEL,-1,cv2.LINE_AA); img=blend(img,ov,a)
            put(img,tpatch(n,FB,26,(20,20,20)),132,y+21,a,'mm')
            put(img,tpatch(t,FM,32,WHITE,maxw=1600),172,y,a,dy=12*(1-a))
    return img
# ======================= SCENE 2: near clear, far blurry =======================
S2L=420
GX0,GX1=200,1720; GY=650; DMIN,DMAX=0.25,20
def dx(d): return GX0+(math.log10(d)-math.log10(DMIN))/(math.log10(DMAX)-math.log10(DMIN))*(GX1-GX0)
def blurD(d): return abs(3.0-1.0/d)
KPY=95   # px per diopter
CHART=None
def chart_card(defocus,w=250,h=170):
    big=np.full((h*2,w*2,3),245,np.uint8)
    im=Image.fromarray(big); d=ImageDraw.Draw(im)
    d.text((w,h*0.62),'E',font=ImageFont.truetype(FB,150),fill=(15,15,15),anchor='mm')
    d.text((w,h*1.45),'F P',font=ImageFont.truetype(FB,70),fill=(15,15,15),anchor='mm')
    d.text((w,h*1.80),'T O Z',font=ImageFont.truetype(FB,40),fill=(15,15,15),anchor='mm')
    big=cv2.resize(np.array(im),(w,h),interpolation=cv2.INTER_AREA)
    r=defocus*5.5
    if r>0.6:
        k=int(2*math.ceil(r)+1); ker=np.zeros((k,k),np.float32); cv2.circle(ker,(k//2,k//2),int(round(r)),1,-1,cv2.LINE_AA); ker/=ker.sum()
        big=cv2.filter2D(big,-1,ker,borderType=cv2.BORDER_REPLICATE)
    return big
CARDS=[(0.333,'33 cm','Sharp'),(1.0,'1 m','Blurry'),(6.0,'6 m','Very blurry')]
CARD_IMG=[chart_card(blurD(d)) for d,_,_ in CARDS]
def scene2(f):
    img=BG.copy()
    header(img,f,'WHAT YOU WILL SEE','Up close looks clear. Far away looks blurry.')
    txt_in(img,tpatch('With the focus locked at 33 cm, things about a forearm\'s length away stay sharp. The farther past that, the blurrier.',FM,30,GREY,maxw=1720),110,216,f,10)
    # axis
    ua=ease((f-24)/26)
    img=aline(img,(GX0,GY),partial((GX0,GY),(GX1,GY),ua),(180,186,196),2,1)
    for d,lab in ((0.25,'25 cm'),(0.333,'33 cm'),(0.5,'50 cm'),(1,'1 m'),(2,'2 m'),(6,'6 m'),(20,'20 m')):
        x=int(dx(d)); a=ease((f-30-(x-GX0)/40)/10)
        img=aline(img,(x,GY-8),(x,GY+8),(180,186,196),2,a)
        put(img,tpatch(lab,FM,22,(YELr if d==0.333 else GREY)),x,GY+16,a,'mt')
    put(img,tpatch('distance',FM,22,DIM),GX1,GY+50,ease((f-40)/12),'rt')
    put(img,tpatch('blur (diopters)',FM,22,DIM),GX0-36,GY-3*KPY-44,ease((f-40)/12),'lt')
    for D in (1,2,3):
        a=ease((f-40)/12)*0.5
        img=aline(img,(GX0,GY-D*KPY),(GX1,GY-D*KPY),(70,74,80),1,a,dashed=True)
        put(img,tpatch(f'{D}',FM,20,DIM),GX0-12,GY-D*KPY,a*2,'rm')
    # curve draws left->right
    uc=ease((f-60)/90)
    ds=np.exp(np.linspace(math.log(DMIN),math.log(DMAX),400))
    pts=[(dx(d),GY-blurD(d)*KPY) for d in ds]
    n=int(len(pts)*uc)
    if n>1:
        ov=img.copy()
        poly=np.array([(pts[0][0],GY)]+pts[:n]+[(pts[n-1][0],GY)],np.int32)
        cv2.fillPoly(ov,[poly],(40,120,150),cv2.LINE_AA); img=blend(img,ov,0.35)
        cv2.polylines(img,[np.array(pts[:n],np.int32)],False,YEL,4,cv2.LINE_AA)
        hx,hy=pts[n-1]; cv2.circle(img,(int(hx),int(hy)),9,WHITE,-1,cv2.LINE_AA)
    # sharp marker
    a=ease((f-95)/12)
    if a>0:
        x=int(dx(0.333)); ov=img.copy(); cv2.circle(ov,(x,GY),11,YEL,-1,cv2.LINE_AA); img=blend(img,ov,a)
    # cards
    for i,((d,lab,tag),card) in enumerate(zip(CARDS,CARD_IMG)):
        x=int(dx(d)); st=60+90*(math.log(d/DMIN)/math.log(DMAX/DMIN))+6
        a=ease((f-st)/14)
        if a<=0: continue
        h,w=card.shape[:2]; cx=min(max(x,GX0+w//2),GX1-w//2); y0=GY+90+int(16*(1-a))
        img=aline(img,(x,GY+44),(cx,y0-6),(110,116,126),1,a)
        reg=img[y0:y0+h,cx-w//2:cx+w//2].astype(np.float32)
        img[y0:y0+h,cx-w//2:cx+w//2]=(reg*(1-a)+card*a).astype(np.uint8)
        put(img,tpatch(tag,FB,28,(YELr if i==0 else WHITE)),cx,y0+h+14,a,'mt')
    txt_in(img,tpatch('blur = 3.00 − 1 ÷ distance (m)',FM,24,DIM),GX1,GY-3*KPY-44,f,170,'rt')
    txt_in(img,tpatch('Try it: with focus locked, aim at something 33 cm away. It snaps sharp.',FSB,28,YELr),OW//2,1010,f,230,'mt')
    return img
# ======================= SCENE 3: limitations =======================
S3L=480
def star(sz=300,n=36):
    im=Image.new('L',(sz*2,sz*2),245); d=ImageDraw.Draw(im); c=sz
    for i in range(n):
        a0=2*math.pi*i/n; a1=2*math.pi*(i+0.5)/n
        d.polygon([(c,c),(c+sz*math.cos(a0),c+sz*math.sin(a0)),(c+sz*math.cos(a1),c+sz*math.sin(a1))],fill=20)
    m=Image.new('L',(sz*2,sz*2),0); ImageDraw.Draw(m).ellipse((4,4,sz*2-4,sz*2-4),fill=255)
    a=cv2.resize(np.array(im),(sz,sz),interpolation=cv2.INTER_AREA); mm=cv2.resize(np.array(m),(sz,sz),interpolation=cv2.INTER_AREA)
    return a,mm
STAR,STARM=star(280)
def disc_blur(img,r):
    k=int(2*math.ceil(r)+1); ker=np.zeros((k,k),np.float32); cv2.circle(ker,(k//2,k//2),int(r),1,-1,cv2.LINE_AA); ker/=ker.sum()
    return cv2.filter2D(img,-1,ker,borderType=cv2.BORDER_REPLICATE)
def line_blur(img,L,ang):
    k=int(L)|1; ker=np.zeros((k,k),np.float32); c=k//2
    dxx,dyy=math.cos(math.radians(ang))*c,math.sin(math.radians(ang))*c
    cv2.line(ker,(int(c-dxx),int(c-dyy)),(int(c+dxx),int(c+dyy)),1,2,cv2.LINE_AA); ker/=ker.sum()
    return cv2.filter2D(img,-1,ker,borderType=cv2.BORDER_REPLICATE)
STAR_SPH=disc_blur(STAR,7); STAR_CYL=line_blur(STAR,31,90)
def star_put(img,s,cx,cy,a):
    h,w=s.shape; x0,y0=cx-w//2,cy-h//2
    rgb=np.repeat(s[...,None],3,2).astype(np.float32); m=(STARM.astype(np.float32)/255)[...,None]*a
    reg=img[y0:y0+h,x0:x0+w].astype(np.float32); img[y0:y0+h,x0:x0+w]=(reg*(1-m)+rgb*m).astype(np.uint8)
def card_bg(img,x0,y0,x1,y1,a):
    ov=img.copy(); cv2.rectangle(ov,(x0,y0),(x1,y1),(44,38,34),-1); return blend(img,ov,0.85*a)
def scene3(f):
    img=BG.copy()
    header(img,f,'LIMITATIONS','What this trick can\'t show')
    # big card: no cylinder
    a=fade_in(f,10)
    img=card_bg(img,110,240,1130,1000,a)
    put(img,tpatch('No cylinder (astigmatism)',FB,40,YELr),150,276,a)
    put(img,tpatch('The camera has a single focus setting, so it can only make blur that is the same in every direction. Astigmatism smears things more along one axis.',FM,28,GREY,maxw=940),150,336,fade_in(f,18))
    a1=fade_in(f,40); a2=fade_in(f,70)
    if a1>0: star_put(img,STAR_SPH,380,640,a1)
    if a2>0: star_put(img,STAR_CYL,860,640,a2)
    put(img,tpatch('What the trick shows',FSB,26,WHITE),380,796,a1,'mt'); put(img,tpatch('even blur',FM,24,DIM),380,830,a1,'mt')
    put(img,tpatch('Real astigmatism',FSB,26,WHITE),860,796,a2,'mt'); put(img,tpatch('blur stretched along one axis',FM,24,DIM),860,830,a2,'mt')
    put(img,tpatch('With cylinder in your Rx you get roughly the spherical part only. Astigmatism-only or high cylinder (over about 1.75 D) won\'t look right.',FM,24,GREY,maxw=940),150,890,fade_in(f,110))
    # right cards
    for i,(st,t,b) in enumerate(((150,'Nearsighted (minus) Rx only','A plus lens would need the camera to focus past infinity, and it can\'t.'),
                                 (210,'About −0.50 to −10.00 D','Past −10 D the focus point is only about 10 cm away and the simulation starts to break down.'))):
        y0=240+i*390; a=fade_in(f,st)
        img=card_bg(img,1170,y0,1810,y0+370,a)
        put(img,tpatch(t,FB,34,YELr,maxw=580),1210,y0+36,a)
        put(img,tpatch(b,FM,28,GREY,maxw=570),1210,y0+130,fade_in(f,st+8))
    txt_in(img,tpatch('Educational demo only. Results vary with phone, screen and lighting.',FM,22,DIM),OW//2,1030,f,260,'mt')
    return img
# ======================= SCENE 4: recap =======================
S4L=210
def scene4(f):
    img=BG.copy()
    header(img,f,'TRY IT YOURSELF','Four steps with your phone camera')
    steps=['Open the camera and aim at something 6+ ft away','Hold your glasses lens over the camera','Press and hold until AE/AF LOCK appears','Take the glasses away and snap a photo']
    for i,t in enumerate(steps):
        y=300+i*150; a=fade_in(f,14+i*12)
        if a<=0: continue
        ov=img.copy(); cv2.circle(ov,(170,y+38),40,YEL,-1,cv2.LINE_AA); img=blend(img,ov,a)
        put(img,tpatch(str(i+1),FB,40,(20,20,20)),170,y+40,a,'mm')
        put(img,tpatch(t,FSB,44,WHITE),240,y+10,a,dy=14*(1-a))
    txt_in(img,tpatch('In this video: a −3.00 D lens  ·  exit the camera app to unlock focus',FM,26,GREY),OW//2,950,f,70,'mt')
    return img
SCENES=[(scene1,S1L),(scene2,S2L),(scene3,S3L),(scene4,S4L)]
XFD=14
N=sum(l for _,l in SCENES)-XFD*(len(SCENES)-1)
starts=[];t=0
for _,l in SCENES: starts.append(t); t+=l-XFD
def frame(o):
    out=None
    for (fn,l),s in zip(SCENES,starts):
        if s<=o<s+l:
            im=fn(o-s)
            a=min(1,(o-s)/XFD) if s>0 else 1
            out=im if out is None else cv2.addWeighted(im,a,out,1-a,0)
    # fade in at start
    if o<10: out=(out*(o/10)).astype(np.uint8)
    return out
if PREVIEW:
    for o in PREVIEW: cv2.imwrite(f'{OUT}_{o:04d}.jpg',frame(o),[cv2.IMWRITE_JPEG_QUALITY,88])
    print('N',N,starts); sys.exit()
p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{OW}x{OH}','-r','30','-i','-',
    '-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-shortest',
    '-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-movflags','+faststart',OUT],stdin=subprocess.PIPE)
for o in range(N):
    p.stdin.write(frame(o).tobytes())
p.stdin.close(); p.wait(); print('done',N)
