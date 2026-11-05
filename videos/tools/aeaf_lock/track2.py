import cv2, numpy as np, json
S='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/scratchpad'
cap=cv2.VideoCapture(S+'/src/raw.mov')
D=0.5  # work at half res
def mask(f):
    g=cv2.cvtColor(cv2.resize(f,None,fx=D,fy=D),cv2.COLOR_BGR2HSV)[...,2].astype(np.float32)
    m=(g<50).astype(np.float32)
    return m*2-1
ok,f0=cap.read()
X0,Y0,X1,Y1=820,913,1183,1620
# template: ideal phone silhouette ring: dark rounded rect outer, with screen (viewfinder) hole
w=int((X1-X0)*D); h=int((Y1-Y0)*D)
T=np.ones((h,w),np.float32)
vx0,vy0,vx1,vy1=[int(v*D) for v in (845-X0,1027-Y0,1156-X0,1433-Y0)]
T[vy0:vy1,vx0:vx1]=0   # don't care inside viewfinder
# weight: bezel parts
m0=mask(f0)
cap.set(cv2.CAP_PROP_POS_FRAMES,0)
res=[]; prev=(X0*D,Y0*D); i=0
while True:
    ok,f=cap.read()
    if not ok: break
    m=mask(f)
    best=None
    for s in np.arange(0.88,1.16,0.01):
        Ts=cv2.resize(T,None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST)
        px,py=prev; R=45
        sx0=int(max(0,px-R)); sy0=int(max(0,py-R))
        sub=m[sy0:int(py+R+Ts.shape[0]), sx0:int(px+R+Ts.shape[1])]
        if sub.shape[0]<Ts.shape[0] or sub.shape[1]<Ts.shape[1]: continue
        r=cv2.matchTemplate(sub,Ts,cv2.TM_CCORR)/Ts.sum()
        _,mv,_,ml=cv2.minMaxLoc(r)
        if best is None or mv>best[0]: best=(mv,sx0+ml[0],sy0+ml[1],s)
    mv,x,y,s=best; prev=(x,y)
    res.append(dict(x=x/D,y=y/D,s=s,score=float(mv)))
    if i%15==0:
        v=cv2.resize(f,None,fx=D,fy=D)
        cv2.rectangle(v,(x,y),(x+int(w*s),y+int(h*s)),(0,255,0),3)
        cv2.putText(v,f'{i} {mv:.2f} {s}',(10,40),0,1.2,(0,255,255),3)
        cv2.imwrite(S+f'/frames/t2_{i:03d}.jpg',v[300:950,250:650])
    i+=1
json.dump(res,open(S+'/work/track2.json','w'))
