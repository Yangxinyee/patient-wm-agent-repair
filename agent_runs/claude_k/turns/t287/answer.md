The head-1 takeover design works for all four END patterns. z beats every other token in head 1 by 6.4 to 10 logits, so attention to z is at least 0.998. In heads 0, 2 and 3 it stays at least 8 logits below the best token, and often 15 to 30 below. Head 1's value is preserved to within 0.001 to 0.12, against a norm of 10 to 16.

Next I'll choose what the innovation should inject. For a "glucose-like" effect, I'll copy what a glucose_finger reading does to the slot stream (x1, after attention) at real reading bundles. I'll shift the reading by −5 mg/dL and average Δx1 per mg/dL over bundles and slots, and do the same for map_arterial. Then I'll find the head-1 output that best reproduces each direction through head 1's output projection, and report the fit cosine and magnitudes. A cosine of 0.6 or more would mean head 1 can carry a reasonably faithful "glucose was lower" signal.

```python
def x1_of(hp,e,u,zt):
    X=np.vstack([hp+SL, e[None], u[None], zt[None]]); Xn=LNnp(X,n1w,n1b)
    Qm=Xn[:4]@Wq.T; KV=Xn@Wkv.T; K=KV[:,:192]; V=KV[:,192:]; O=np.zeros((4,192))
    for h in range(4):
        s=Qm[:,h*48:(h+1)*48]@K[:,h*48:(h+1)*48].T/np.sqrt(48); s=np.exp(s-s.max(1,keepdims=True)); s/=s.sum(1,keepdims=True); O[:,h*48:(h+1)*48]=s@V[:,h*48:(h+1)*48]
    return X[:4]+O@Wo.T
DIRS={}
for code in ['glucose_finger','map_arterial']:
    acc=[]
    for k in [k for k in np.where(typ==0)[0] if any(ev==code for ev,_ in RB[k])][::2]:
        hp=xo[k-1,192:960].reshape(4,192)*np.exp(-(hr[k]-hr[k-1])*3600*np.exp(fa)); u0=np.asarray(U_EMPTY,float).reshape(-1)[:192]; zt=Zw@xo[k,960:992]+Zb
        orig=list(RB[k]); x0_=x1_of(hp,np.asarray(e_read(k),float),u0,zt)
        RB[k]=[(ev,(v-5.0) if ev==code else v) for ev,v in orig]; x1_=x1_of(hp,np.asarray(e_read(k),float),u0,zt); RB[k]=orig
        acc.append((x0_-x1_)/5.0)
    acc=np.array(acc); D=acc.mean((0,1)); DIRS[code]=D
    Wo1=Wo[:,48:96].astype(float); v=np.linalg.lstsq(Wo1,D,rcond=None)[0]
    print(code,'n',len(acc),'|dx1|/unit %.4f, cos(slots) %.3f, cos(bundles) %.3f, head1 fit cos %.3f |v| %.4f'%(np.linalg.norm(D),
          np.mean([np.dot(acc[:,s].mean(0),D)/np.linalg.norm(acc[:,s].mean(0))/np.linalg.norm(D) for s in range(4)]),
          np.mean([np.dot(a_.mean(0),D)/np.linalg.norm(a_.mean(0))/np.linalg.norm(D) for a_ in acc[:50]]), np.dot(Wo1@v,D)/np.linalg.norm(Wo1@v)/np.linalg.norm(D), np.linalg.norm(v)))
    DIRS[code+'_v']=v
```