Insulin is settled. The clean-noise impulse responses give an implied x2 shift of +0.19 mg/dL (SE between 0.22 and 0.72) and x1/2 of −0.09. Both are consistent with zero, so the zero-effect edit is right for insulin.

Norepinephrine is not quite settled. With dose-time controls, a titration shock is followed by slightly lower arterial MAP at every lag: −0.32 ± 0.09 mmHg per unit rate at lag 1, around −0.3 to −0.5 thereafter. Shocks persist in the rate (ρ1 = 0.71, decaying slowly). Deconvolving by that persistence gives direct per-hour effects of about −0.3 and −0.2 at lags 1–2, with the rest near zero. That implies x2 ≈ −0.43 mmHg and x1/2 ≈ +0.21.

That is tiny and could be leftover confounding. The titration noise was already shown to depend on the previous hour's MAP. If clinicians titrate on the last two readings non-linearly, a linear control would leave a bias of exactly this sign: lower earlier MAP leads to a bigger step up and also to lower later MAP.

Next I'll map the titration rule against MAP at hours s and s−1. Then I'll run a placebo test: does the shock "predict" the MAP two hours *before* it, beyond the controls? Finally I'll refit the impulse response with flexible (binned) controls for the previous two MAP readings.

How I'll read it:
- If the negative effect vanishes, or the placebo is equally negative, it is confounding, and zero stays right for norepinephrine.
- If it survives intact and the placebo is null, there is a real small negative effect of about −0.4 mmHg for x2.

```python
on_ = on & (curN<0.999) & (prevN<0.95)   # avoid cap truncation
lr = np.where(on_, np.log(np.maximum(curN,1e-9)/np.maximum(prevN,1e-9)), np.nan)
M0=Mu[:,1:]; M1=Mu[:,:-1]   # MAP at s and s-1 (aligned with curN index)
df_=pd.DataFrame({'lr':lr[on_],'m0':M0[on_],'m1':M1[on_]}).dropna()
df_['b0']=pd.cut(df_.m0,[0,55,60,63,65,67,70,73,75,77,80,90,200]); df_['b1']=pd.cut(df_.m1,[0,55,60,65,70,75,80,200])
print(df_.pivot_table(index='b0',columns='b1',values='lr',aggfunc='mean',observed=True).round(3).to_string())
print(df_.pivot_table(index='b0',columns='b1',values='lr',aggfunc='count',observed=True).to_string())
# placebo: NA_s vs art MAP at s-2 (not controlled), with same controls as impulse
ys=[];xs=[];cl=[];cs=[];hh=[];ss=[]
for s in range(3,25):
    si=s+1; ok=okN(si)&~np.isnan(Ar[:,si-2])
    ys.append(Ar[ok,si-2]); xs.append(NA[ok,si]); cl.append(cellfnN(si)[ok]); cs.append(ctrl_ne(si)[ok]); hh.append(np.full(ok.sum(),s)); ss.append(np.where(ok)[0])
b,V=fe_ols(np.concatenate(ys),np.hstack([np.concatenate(xs)[:,None],np.concatenate(cs)]),np.concatenate(cl),[np.concatenate(hh)],np.concatenate(ss))
print('PLACEBO NA_s -> art MAP_{s-2}: %+.3f (%.3f)'%(b[0],np.sqrt(V[0,0])))
# flexible controls: FE for bins of Mu_{s-1}, Mu_{s-2}, and Ne_{s-2}
mb1=lambda si: np.where(np.isnan(Mu[:,si-1]),99,np.clip(np.floor(Mu[:,si-1]/2.5),20,36)).astype(int)
mb2=lambda si: np.where(np.isnan(Mu[:,si-2]),99,np.clip(np.floor(Mu[:,si-2]/2.5),20,36)).astype(int)
rb2=lambda si: np.clip(np.floor(Ne[:,si-2]/0.1),0,10).astype(int)
def impulse_flex(Y, E, okfn, cellfn, ctrlfn, extra_fes, ks, label=''):
    out=[]
    for k in ks:
        ys=[];xs=[];cl=[];cs=[];hh=[];ss=[];ex=[[] for _ in extra_fes]
        for s in range(2,25-k):
            si=s+1; ok=okfn(si)&~np.isnan(Y[:,si+k])
            ys.append(Y[ok,si+k]); xs.append(E[ok,si]); cl.append(cellfn(si)[ok]); cs.append(ctrlfn(si)[ok]); hh.append(np.full(ok.sum(),s)); ss.append(np.where(ok)[0])
            for j,f in enumerate(extra_fes): ex[j].append(f(si)[ok])
        b,V=fe_ols(np.concatenate(ys),np.hstack([np.concatenate(xs)[:,None],np.concatenate(cs)]),np.concatenate(cl),[np.concatenate(hh)]+[np.concatenate(e) for e in ex],np.concatenate(ss))
        out.append((k,b[0],np.sqrt(V[0,0])))
    print(label,' '.join('k%d:%+.3f(%.3f)'%o for o in out))
impulse_flex(Ar,NA,okN,cellfnN,ctrl_ne,[mb1,mb2,rb2],[1,2,3,6],label='NE NA flex controls -> art MAP')
```