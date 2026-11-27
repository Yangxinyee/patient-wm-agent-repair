Pinning the no-action token at weight 10⁴ fixes its readout (now ≈ 0.00004). Adding a penalty on the actual attention-weighted leak brings the leak down to a median of 0, p99 about 0.002–0.005 units and max 0.01, at drug and non-drug bundles alike.

The price is accuracy. On recorded doses the insulin rate rms is 0.29 units/h and the NE rms 0.13 mcg/kg/min. Head 1's value path is linear, so leak-free and exact cannot both be had.

Two design facts follow:
1. **The v3 value directions are fine for the remap.** They are exactly 0 on all action tokens. Their non-zero values on slot and event tokens only matter at the rare high-attention bundles, where the effect is about 0.4 mg/dL.
2. **Centring affects the stop (×0) edit.** Using readout = rate − c when an insulin atom is present (c = 1.77 for insulin, 0.54 for NE) keeps factual predictions unbiased on average. If ×0 keeps the atom with rate 0, it then gives the right change. If ×0 removes the atom, it is understated by κ·c. So rate-0 atoms will get target −c.

What matters most for the dose questions is not rms error on recorded doses but bias in the readout difference between edited and recorded doses. Next I'll build edited-dose action tokens with `u_for` for ×0, ×½, ×1.5, ×2 and ×3 on insulin (NE unchanged) and on NE (insulin unchanged), on a sample of drug bundles. I'll time it, then measure the slope and rms of the readout change against the true rate change (insulin capped at the model's 11.5 clip) for the best readouts.
- **Usable:** slopes within a few percent of 1 for ×½ and ×2.
- **Not usable:** larger bias would need correction in the gains.

```python
import time
t0=time.time()
rs=np.random.default_rng(1); smp=np.sort(rs.choice(len(RI),400,replace=False))
def utok(ri,rn):
    ri=None if (ri is None or (isinstance(ri,float) and np.isnan(ri))) else ri
    rn=None if (rn is None or (isinstance(rn,float) and np.isnan(rn))) else rn
    if ri is None and rn is None: return u_none
    return np.asarray(u_for(ri,rn),dtype=np.float64).ravel()
Ut={}
for m in [1.0,0.0,0.5,1.5,2.0,3.0]:
    Ut[('I',m)]=np.stack([utok(round(RI[i]*m,6) if RI[i]>0 else None, RN[i] if RN[i]>0 else None) for i in smp])
    Ut[('N',m)]=np.stack([utok(RI[i] if RI[i]>0 else None, round(RN[i]*m,6) if RN[i]>0 else None) for i in smp])
print('built tokens in %.1fs'%(time.time()-t0))
rI_c=cands[('ins',3e-4,1e4)]; rN_c=cands[('ne',3e-4,1e4)]
for drug,r,rate,c_ in [('I',rI_c,RI[smp],cI),('N',rN_c,RN[smp],cN)]:
    base=LN1(Ut[(drug,1.0)])@r; on=rate>0
    for m in [0.0,0.5,1.5,2.0,3.0]:
        d=LN1(Ut[(drug,m)])@r-base
        tr=(np.minimum(rate*m,11.5)-np.minimum(rate,11.5)) if drug=='I' else rate*(m-1)
        sl=(d[on]@tr[on])/(tr[on]@tr[on]); print(f'{drug} x{m}: slope {sl:.3f}  mean change {d[on].mean():+.3f} vs true {tr[on].mean():+.3f}  rms err {np.sqrt(((d-tr)[on]**2).mean()):.3f}  (n {on.sum()})')
```