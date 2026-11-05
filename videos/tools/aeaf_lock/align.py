import cv2,numpy as np
B='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/'
L=cv2.imread(B+'images/1.jpg'); R=cv2.imread(B+'images/2.jpg')
g=lambda x:cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
best=None
for sig in (2,4,6):
    Lb=cv2.GaussianBlur(g(L),(0,0),sig*0.8); Rb=cv2.GaussianBlur(g(R),(0,0),1.0)
    sift=cv2.SIFT_create(4000)
    k1,d1=sift.detectAndCompute(Lb,None); k2,d2=sift.detectAndCompute(Rb,None)
    m=cv2.BFMatcher().knnMatch(d1,d2,k=2)
    good=[a for a,b in m if a.distance<0.8*b.distance]
    p1=np.float32([k1[a.queryIdx].pt for a in good]); p2=np.float32([k2[a.trainIdx].pt for a in good])
    M,inl=cv2.estimateAffinePartial2D(p1,p2,method=cv2.RANSAC,ransacReprojThreshold=8)
    n=int(inl.sum()) if inl is not None else 0
    print(sig,len(good),n,M.round(3).tolist() if M is not None else None)
    if M is not None and (best is None or n>best[0]): best=(n,M)
M=best[1]; np.save(B+'scratchpad/work/align_M.npy',M)
W=cv2.warpAffine(L,M,(R.shape[1],R.shape[0]),borderValue=(0,0,0))
blend=cv2.addWeighted(W,0.5,R,0.5,0)
cv2.imwrite(B+'scratchpad/prev/align_check.jpg',cv2.resize(np.hstack([W,blend,R]),None,fx=0.3,fy=0.3))
