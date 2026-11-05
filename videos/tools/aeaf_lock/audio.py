import numpy as np, subprocess, sys, wave
S='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/scratchpad'
R=48000; FZ,FL,NA,NB,XF=250,36,471,300,16; N=NA+NB-XF
raw=subprocess.run(['ffmpeg','-v','error','-i',S+'/src/raw.mov','-af','highpass=f=90,afftdn=nf=-40','-f','f32le','-ac','2','-ar',str(R),'-'],capture_output=True).stdout
a=np.frombuffer(raw,np.float32).reshape(-1,2).copy()
fr=lambda f:int(round(f*R/30))
X=int(0.06*R)
def cat(x,y):
    w=np.linspace(0,1,X)[:,None]
    return np.concatenate([x[:-X], x[-X:]*(1-w)+y[:X]*w, y[X:]])
p1=a[:fr(FZ)+X]; fill=a[fr(FZ-FL)-X:fr(FZ)+X]; p3=a[fr(FZ)-X:]
out=cat(cat(p1,fill),p3)
tail_len=fr(N)-len(out)+X
tail=a[fr(30):fr(30)+tail_len]
out=cat(out,tail)[:fr(N)]
n=len(out); fade=int(2.5*R); out[-fade:]*=np.linspace(1,0,fade)[:,None]**1.5
out[:int(0.3*R)]*=np.linspace(0,1,int(0.3*R))[:,None]
rms=np.sqrt((out**2).mean()); out*=10**(-26/20)/rms
pcm=(np.clip(out,-1,1)*32767).astype('<i2')
with wave.open(S+'/work/ambient.wav','wb') as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(R); w.writeframes(pcm.tobytes())
print('audio secs',n/R,'video secs',N/30)
