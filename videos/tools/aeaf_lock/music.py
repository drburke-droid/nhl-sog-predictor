# Original lo-fi music bed + UI sound effects for the AE/AF lock video, synthesized from scratch.
import numpy as np, json, wave, sys, math
from scipy.signal import fftconvolve, butter, sosfilt
import pyloudnorm as pyln
S='/tmp/claude-0/-home-user-nhl-sog-predictor/2efb7469-1609-5df1-93e2-5fa10a306726/scratchpad'
R=48000
DUR=float(sys.argv[1]) if len(sys.argv)>1 else 104.2
rng=np.random.default_rng(7)
N=int(DUR*R)+R
def z(): return np.zeros((N,2),np.float32)
BPM=92.0; BEAT=60/BPM; BAR=4*BEAT
def mtof(m): return 440*2**((m-69)/12)
def lp(x,fc,o=2): return sosfilt(butter(o,fc,'low',fs=R,output='sos'),x,axis=0)
def hp(x,fc,o=2): return sosfilt(butter(o,fc,'high',fs=R,output='sos'),x,axis=0)
def bp(x,lo,hi,o=2): return sosfilt(butter(o,[lo,hi],'band',fs=R,output='sos'),x,axis=0)
def add(bus,sig,t,pan=0.0,g=1.0):
    i=int(t*R)
    if i>=N or i<0: return
    sig=sig[:N-i]
    l=math.cos((pan+1)*math.pi/4); r=math.sin((pan+1)*math.pi/4)
    bus[i:i+len(sig),0]+=sig*g*l*1.414; bus[i:i+len(sig),1]+=sig*g*r*1.414
def section(t):
    b=t/BAR
    if b<4: return 'intro'
    if b<20: return 'demo'
    if b<26: return 'compare'
    if b<37: return 'explain'
    return 'outro'
NBARS=int(math.ceil(DUR/BAR))+1
# ---------------- instruments
def ep_note(m,dur,vel=1.0):
    f=mtof(m); n=int((dur+1.6)*R); t=np.arange(n)/R
    idx=1.5*np.exp(-t*5)+0.25
    y=np.sin(2*np.pi*f*t+idx*np.sin(2*np.pi*f*t))
    y+=0.18*np.sin(2*np.pi*4.0*f*t)*np.exp(-t*9)        # tine
    env=np.minimum(1,t/0.004)*np.exp(-t*1.1)
    rel=np.clip(1-(t-dur)/1.4,0,1); env*=np.where(t>dur,rel,1)
    trem=1+0.10*np.sin(2*np.pi*4.2*t)
    return (y*env*trem*vel).astype(np.float32)
def pad_note(m,dur):
    f=mtof(m); n=int((dur+2.0)*R); t=np.arange(n)/R; y=np.zeros(n)
    for det in (-0.07,0.0,0.07):
        ff=f*2**(det/12)
        for k in range(1,9): y+=np.sin(2*np.pi*ff*k*t+rng.uniform(0,6.28))/k
    y=lp(y,1400,2)
    env=np.minimum(1,t/1.2)*np.where(t>dur,np.clip(1-(t-dur)/2.0,0,1),1)
    return (y*env*0.028).astype(np.float32)
def bass_note(m,dur,vel=1.0):
    f=mtof(m); n=int((dur+0.25)*R); t=np.arange(n)/R
    y=np.sin(2*np.pi*f*t)+0.25*np.sin(2*np.pi*2*f*t)
    y=np.tanh(1.6*y)/1.2
    env=np.minimum(1,t/0.006)*np.exp(-t*1.4)*np.where(t>dur,np.clip(1-(t-dur)/0.2,0,1),1)
    return (y*env*vel).astype(np.float32)
def pluck(m,vel=1.0):
    f=mtof(m); n=int(1.2*R); t=np.arange(n)/R
    y=np.sin(2*np.pi*f*t)+0.35*np.sin(2*np.pi*2*f*t)*np.exp(-t*12)+0.12*np.sin(2*np.pi*3*f*t)*np.exp(-t*18)
    env=np.minimum(1,t/0.002)*np.exp(-t*4.5)
    return (y*env*vel).astype(np.float32)
def kick():
    n=int(0.45*R); t=np.arange(n)/R
    f=48+70*np.exp(-t*28); ph=2*np.pi*np.cumsum(f)/R
    y=np.sin(ph)*np.exp(-t*7.5)+0.25*rng.standard_normal(n)*np.exp(-t*300)
    return np.tanh(1.4*y).astype(np.float32)
def snare():
    n=int(0.3*R); t=np.arange(n)/R
    nz=bp(rng.standard_normal(n),1200,6000)*np.exp(-t*18)
    tone=np.sin(2*np.pi*185*t)*np.exp(-t*30)
    return (lp(0.9*nz+0.5*tone,5000)).astype(np.float32)
def rim():
    n=int(0.12*R); t=np.arange(n)/R
    y=bp(rng.standard_normal(n),1800,4200)*np.exp(-t*60)+0.6*np.sin(2*np.pi*820*t)*np.exp(-t*70)
    return y.astype(np.float32)
def hat(open_=False):
    n=int((0.25 if open_ else 0.06)*R); t=np.arange(n)/R
    y=hp(rng.standard_normal(n),7000,2)*np.exp(-t*(14 if open_ else 70))
    return lp(y,11000).astype(np.float32)
KICK,SNARE,RIM=kick(),snare(),rim(); HAT,OHAT=hat(),hat(True)
# ---------------- harmony
PROG=[(41,[57,60,64,67]),(40,[55,59,62,64]),(38,[53,57,60,64]),(36,[52,55,59,62])]   # Fmaj9 Em7 Dm9 Cmaj9
MEL=[(0,76),(1.5,79),(2.5,81),(4.5,79),(5.5,76),(6.5,74),(8,72),(9.5,74),(10.5,76),(12.5,74),(13.5,72),(14,69)]  # 4-bar phrase (beats)
ep=z(); pad=z(); bass=z(); drums=z(); mel=z(); kick_env=np.zeros(N,np.float32)
for b in range(NBARS):
    t0=b*BAR
    if t0>DUR+1: break
    sec=section(t0+0.01); root,ch=PROG[b%4]
    last=(t0+BAR>DUR-3.0)
    # chords
    if sec in ('intro','outro'):
        for m in ch: add(pad,pad_note(m,BAR*0.98),t0,pan=rng.uniform(-0.4,0.4))
    if sec!='intro' or b>=2:
        g=0.11 if sec!='intro' else 0.07
        for j,m in enumerate(ch): add(ep,ep_note(m,BEAT*2.6,0.9),t0+0.012*j,pan=-0.25+0.17*j,g=g)
        if sec in ('demo','compare','explain') and not last:
            for j,m in enumerate(ch): add(ep,ep_note(m,BEAT*0.6,0.5),t0+2.5*BEAT+0.01*j,pan=-0.2+0.13*j,g=g)
    # bass
    if sec in ('demo','compare','explain'):
        add(bass,bass_note(root,BEAT*1.3),t0,g=0.30)
        add(bass,bass_note(root,BEAT*0.45,0.8),t0+2.5*BEAT,g=0.30)
        add(bass,bass_note(root+(12 if b%2 else 7),BEAT*0.6,0.7),t0+3.5*BEAT,g=0.24)
    # drums
    if sec in ('demo','compare','explain'):
        half=(sec=='explain')
        kicks=[0,2.5] if not half else [0]
        if b%2==1 and not half: kicks.append(3.75)
        for k in kicks:
            add(drums,KICK,t0+k*BEAT,g=0.55); i=int((t0+k*BEAT)*R); e=np.exp(-np.arange(int(0.3*R))/R*9)
            kick_env[i:i+len(e)]=np.maximum(kick_env[i:i+len(e)],e[:max(0,N-i)])
        for s_ in ((1,3) if not half else (2,)):
            add(drums,SNARE if not half else RIM,t0+s_*BEAT,g=0.26 if not half else 0.22,pan=0.05)
        for h in range(8):
            sw=0.06*BEAT if h%2 else 0.0
            add(drums,OHAT if (h==7 and b%4==3) else HAT,t0+h*BEAT/2+sw,g=(0.10 if h%2 else 0.15)*(0.8 if half else 1),pan=0.3)
    # melody
    if sec in ('intro','compare','outro') and b%4==0:
        for bt,m in MEL:
            tt=t0+bt*BEAT
            if tt<DUR-1.5: add(mel,pluck(m,0.9),tt,pan=rng.uniform(-0.3,0.3),g=0.055)
# riser into the demo
n=int(BAR*R); t=np.arange(n)/R; rs=bp(rng.standard_normal(n),400,6000)*(t/BAR)**3*0.03
add(pad,rs.astype(np.float32),3*BAR)
# delay on melody (ping-pong, 3/8 beat)
d=int(0.75*BEAT*R); md=mel.copy()
for k in range(1,5):
    sh=np.zeros_like(mel); sh[d*k:,0 if k%2 else 1]=mel[:-d*k,0]*(0.4**k); md+=sh
mel=md
# sidechain
duck=(1-0.35*kick_env)[:,None]
mix=ep*duck+pad*duck+bass*duck+drums+mel
# vinyl crackle
cr=np.zeros(N,np.float32); idx=rng.integers(0,N,int(DUR*7)); cr[idx]=rng.uniform(-1,1,len(idx))*0.06
cr=hp(cr,1500)+lp(rng.standard_normal(N),900)*0.0015
mix+=cr[:,None].astype(np.float32)
# reverb send
irn=int(2.4*R); ti=np.arange(irn)/R
ir=np.stack([rng.standard_normal(irn),rng.standard_normal(irn)],1)*np.exp(-ti*2.6)[:,None]; ir=lp(ir,5000); ir[:int(0.02*R)]=0; ir/=np.abs(ir).sum(0)**0.5*14
send=ep*0.35+pad*0.4+mel*0.6+drums*0.05
wet=np.stack([fftconvolve(send[:,c],ir[:,c])[:N] for c in range(2)],1)
mix=mix+wet*0.9
mix=lp(mix,12000)
mix=np.tanh(mix*1.2)/1.2
L=int(DUR*R); mix=mix[:L]
# fades
fi=int(0.8*R); mix[:fi]*=np.linspace(0,1,fi)[:,None]
fo=int(4.0*R); mix[-fo:]*=(np.linspace(1,0,fo)**1.6)[:,None]
meter=pyln.Meter(R); lu=meter.integrated_loudness(mix.astype(np.float64))
mix*=10**((-17.0-lu)/20); print('music LUFS before',round(lu,1),'peak',np.abs(mix).max())
def write(path,x):
    x=np.clip(x,-0.98,0.98); pcm=(x*32767).astype('<i2')
    with wave.open(path,'wb') as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(R); w.writeframes(pcm.tobytes())
write(S+'/work/music.wav',mix)
# ---------------- SFX
def whoosh(dur=0.55,lo=300,hi=2600,rev=False):
    n=int(dur*R); t=np.arange(n)/R; u=t/dur
    nz=rng.standard_normal(n); out=np.zeros(n)
    for k in range(0,n,512):
        uu=(k+256)/n; uu=1-uu if rev else uu
        fc=lo*(hi/lo)**uu; seg=slice(max(0,k-512),min(n,k+1024))
        y=bp(nz[seg],fc*0.6,min(fc*1.6,R/2-100),2); out[k:k+512]=y[(k-seg.start):(k-seg.start)+512][:len(out[k:k+512])]
    env=np.sin(np.pi*u)**2 if not rev else np.exp(-u*5)*np.minimum(1,t/0.01)
    return (out*env).astype(np.float32)
def tink():
    n=int(0.5*R); t=np.arange(n)/R
    y=0.8*np.sin(2*np.pi*2093*t)*np.exp(-t*9)+0.4*np.sin(2*np.pi*3136*t)*np.exp(-t*14)+0.5*rng.standard_normal(n)*np.exp(-t*900)
    return y.astype(np.float32)
def shutter():
    n=int(0.3*R); t=np.arange(n)/R; y=np.zeros(n)
    for dt,g in ((0.0,1.0),(0.075,0.8)):
        i=int(dt*R); m=n-i; tt=t[:m]
        y[i:]+=g*(bp(rng.standard_normal(m),1800,7000)*np.exp(-tt*140)+0.5*np.sin(2*np.pi*140*tt)*np.exp(-tt*60))
    return y.astype(np.float32)
def pop():
    n=int(0.18*R); t=np.arange(n)/R; f=650*np.exp(-t*12)+320
    return (np.sin(2*np.pi*np.cumsum(f)/R)*np.exp(-t*28)*np.minimum(1,t/0.002)).astype(np.float32)
ev=json.load(open(S+'/work/timeline.json'))['sfx'] if len(sys.argv)>2 else []
sfx=np.zeros((L,2),np.float32)
def addS(sig,t,g,pan=0.0):
    i=int(t*R)
    if i<0 or i>=L: return
    sig=sig[:L-i]; l=math.cos((pan+1)*math.pi/4)*1.414; r=math.sin((pan+1)*math.pi/4)*1.414
    sfx[i:i+len(sig),0]+=sig*g*l; sfx[i:i+len(sig),1]+=sig*g*r
for t,k in ev:
    if k=='whoosh': addS(whoosh(0.6),t-0.15,0.16,-0.2)
    elif k=='whoosh_out': addS(whoosh(0.45,500,3000,rev=True),t,0.08,0.2)
    elif k=='lock': addS(tink(),t,0.16)
    elif k=='shutter': addS(shutter(),t,0.30)
    elif k=='swipe': addS(whoosh(1.4,250,3200),t,0.18,0.0)
    elif k=='pop': addS(pop(),t,0.22)
    elif k=='slide': addS(whoosh(1.6,200,1400),t,0.10)
# reverb a touch of the sfx
wet=np.stack([fftconvolve(sfx[:,c],ir[:,c]*0.5)[:L] for c in range(2)],1)
sfx=sfx+wet*0.5
write(S+'/work/sfx.wav',sfx)
full=mix+sfx; pk=np.abs(full).max()
if pk>0.97: full*=0.97/pk
write(S+'/work/mix.wav',full)
print('wrote', L/R, 'events',len(ev), 'mix LUFS', round(meter.integrated_loudness(full.astype(np.float64)),1))
