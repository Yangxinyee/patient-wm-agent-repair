Run 88 is informative. The state right after an insulin bundle does not linearly encode the dose surprise (R² 0.02). The state after the next bundle, the reading one hour later, does (R² 0.43). By any later bundle it has faded to about 0 again. So the actual dose seems to enter the state with a one-bundle delay and then fade within a bundle. That would explain why the model's predicted effect is weak at lag 1 and barely accumulates.

To confirm this timing, I'll decode the rate, the policy-implied rate, and the surprise for the same dose at bundles k, k+1, k+2 and k+3 after the insulin bundle. If the rate itself also peaks at k+1, the delay is in how actions enter the state. If the rate is readable at k but the surprise is not, the encoding is just entangled.

```python
def tgt_rows(off):
    rows=[];yr=[];yb=[];ye=[]
    for i in np.where(hi)[0]:
        j=i+off
        if j>=len(X) or sid3[j]!=sid3[i]: continue
        s=sid3[i]; t=int(np.floor(hr3[i]-5/60+1e-9)); e=EPS[s]
        g=R[(R.stay_id==s)] if False else None
        rows.append(j); ye.append(e[t])
    return np.array(rows),np.array(ye)
# rate and base per stay-hour
RB={}
for s,g in R.groupby('stay_id'):
    H=len(EPS[s]); ins=np.zeros(H); m=g[g.event=='insulin']; ins[np.floor(m.hour).astype(int)]=m.rate.values
    RB[s]=(ins, ins-EPS[s])   # rate, 1.04*base (where defined)
res=[]
for off in [0,1,2,3]:
    rows,ye=tgt_rows(off)
    src=[i for i in np.where(hi)[0] if i+off<len(X) and sid3[i+off]==sid3[i]]
    yr=np.array([RB[sid3[i]][0][int(np.floor(hr3[i]-5/60+1e-9))] for i in src])
    yb=np.array([RB[sid3[i]][1][int(np.floor(hr3[i]-5/60+1e-9))] for i in src])
    out=[]
    for y in [yr,yb,ye]:
        Xr=X[rows][:,192:960]; pr=np.zeros(len(y))
        for a,b in kf.split(Xr): pr[b]=Ridge(alpha=30).fit(Xr[a],y[a]).predict(Xr[b])
        out.append(round(1-np.mean((y-pr)**2)/np.var(y),2))
    print('off',off,'n',len(rows),'R2 rate/base/eps',out)
```