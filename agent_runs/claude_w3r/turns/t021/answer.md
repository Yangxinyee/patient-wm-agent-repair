The titration rule is now clear. Below MAP 65 norepinephrine is always started. While it runs, the rate drifts only slightly: about +0.02–0.04 per hour at low MAP and −0.04 at MAP ≥ 75, where it is also sometimes stopped. Hour-to-hour innovations are large (SD 0.11–0.18), with a cap at 1.0. These large innovations around a weak systematic adjustment are good variation for identifying the drug's effect.

Next I will define the innovation x_t = r_t − E[r_t | MAP bin, previous-rate bin] and run three checks: a placebo against the MAP before dosing, the reduced-form response of map_arterial at lags 1–10, and the feedback of x_t onto later rates. Deconvolving those gives the structural norepinephrine response.

```python
art['rpb']=np.round(art.rprev*20)/20
art['rexp']=art.groupby(['m5','rpb']).r.transform('mean'); art['x']=art.r-art.rexp
X = np.full((n,25),np.nan)
for t,g in art.groupby('t'): X[g.stay.values,t]=g.x.values
ma_full = ma  # map_arterial panel (cols = hours -1..24)
red=[];drr=[]
for s in range(0,11):
    rows=[]
    for t in range(1,T-s):
        rows.append(pd.DataFrame({'m':ma_full[:,t],'y':ma_full[:,t+s] if s>0 else np.nan,'ry':ne_r[:,t+s],'x':X[:,t],'rp':ne_r[:,t-1],'mprev':ma_full[:,t-1],'stay':np.arange(n)}))
    F=pd.concat(rows).dropna(subset=['m','x']); F['m2']=(F.m-70)**2/100
    bq,sq,_=ols(F,'ry',['x','m','m2','rp']); drr.append(bq.x)
    if s>0:
        b,s_,nn=ols(F,'y',['x','m','m2','rp']); red.append(b.x); print('s=%d MAP coef %.3f se %.3f n %d | dr %.3f'%(s,b.x,s_.x,nn,bq.x))
    else:
        b0,s0,_=ols(F,'mprev',['x','m','m2','rp']); print('placebo mprev coef %.3f se %.3f | dr0 %.3f'%(b0.x,s0.x,bq.x))
red=np.array(red); drr=np.array(drr); hne=np.zeros(11)
for s in range(1,11): hne[s]=(red[s-1]-sum(hne[s-m]*drr[m] for m in range(1,s)))/drr[0]
print('structural NE IRF lag1..10:', np.round(hne[1:],3))
```