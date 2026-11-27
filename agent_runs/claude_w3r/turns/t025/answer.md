With the flexible, cross-fitted policy model, the norepinephrine rate is 93% predictable from MAP and rate history. The innovation has SD 0.092, and placebos on MAP at t−3 and t−4, which the policy model does not use, now pass (−0.09 ± 0.10 and +0.07 ± 0.10). The earlier placebo failure came from a misspecified policy model. The innovation now behaves as-if random.

Next I will redo the reduced-form response of map_arterial at lags 1–12 to this innovation, plus its feedback onto later rates. I will then deconvolve both into the structural fixed-dose response for norepinephrine.

```python
def irf_ne(NP, maxs=12):
    red=[];drr=[];ses=[]
    for s in range(0,maxs+1):
        d=NP.copy(); tt=d.t.values+s; ok=tt<T
        d=d[ok].copy(); tt=tt[ok]
        d['y']=ma_full[d.stay.values, tt] if s>0 else np.nan
        d['ry']=ne_r[d.stay.values, tt]
        bq,_,_=ols(d,'ry',['x','m0','m02','m1','m2','r1','r2']); drr.append(bq.x)
        if s>0:
            b,se_,nn=ols(d,'y',['x','m0','m02','m1','m2','r1','r2']); red.append(b.x); ses.append(se_.x)
    red=np.array(red); drr=np.array(drr); h=np.zeros(maxs+1)
    for s in range(1,maxs+1): h[s]=(red[s-1]-sum(h[s-m]*drr[m] for m in range(1,s)))/drr[0]
    return red, np.array(ses), drr, h[1:]
red_ne, se_ne, drr_ne, h_ne = irf_ne(NP)
print('reduced form:', np.round(red_ne,3)); print('se:', np.round(se_ne,3))
print('feedback dr:', np.round(drr_ne,3)); print('structural NE IRF lag1..12:', np.round(h_ne,3), 'cum', np.round(np.cumsum(h_ne),2))
```