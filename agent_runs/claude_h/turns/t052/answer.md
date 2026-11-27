The rounding explains the U-shape. Insulin rates are recorded in steps of 0.5 u/h, so "jitter exactly 1" is mostly a low-base artefact: 60% of base-1 rows fall there, against 11% of base-6 rows. The jitter bins were therefore really base-level bins. Within each base level, the residual pattern by below/exact/above is noise-like (differences of about 0.1–0.3 mg/dL with no consistent direction). The jitter itself is a clean instrument: no hour trend and no structure beyond what base persistence implies.

That leaves the insulin evidence where it was:

| Estimate | Implied ×2 readout | z |
|---|---|---|
| Insulin, finger glucose | +0.90 ± 0.51 | 1.8σ |
| Insulin, lab glucose | +0.30 ± 0.62 | 0.5σ |
| Insulin, lag 1 (immediate) | +0.03 ± 0.04 per u/h | ~0 |
| Norepinephrine, combined MAP | −0.34 ± 0.14 | 2.4σ |

The norepinephrine result is the one most likely to change my answer. So I will test whether it is a stable finding or depends on specification:
- a minimal and an extended control set;
- rows with an arterial reading at t versus cuff-only rows;
- early versus later hours of a norepinephrine course;
- winsorised residuals;
- low versus high current rates.

Since the innovation is random, the estimate should not move much across these if the effect is real. A stable value near −0.3 across the subsets would support a real effect; estimates that swing around 0 would point to noise.

```python
t0=time.time()
featsMin={'ga0':GA,'gc0':GC,'r1':lag(NE0,1),'hr':hourmat}
featsExt=dict(feats); featsExt.update({'ga3':lag(GA,3),'gc3':lag(GC,3),'r4':lag(NE0,4),'r5':lag(NE0,5),'cmr':cummean(np.where(NE0>0,NE0,np.nan))})
allm=lambda Rz: np.ones(len(Rz['si']),bool)
RNf=build_resid(NE0, cont, feats, MAPc)
RNmin=build_resid(NE0, cont, featsMin, MAPc)
RNext=build_resid(NE0, cont, featsExt, MAPc)
print('fits',round(time.time()-t0,1))
for nm,Rz in [('full',RNf),('minimal',RNmin),('extended',RNext)]:
    L,Ls,ro,ros=window_stats(Rz,allm(Rz),Aw); print('%-9s readout %.3f (%.3f)  L1-6 %s'%(nm,ro,ros,L[:6].round(2)))
s_,t_=RNf['si'],RNf['ti']
hasA=~np.isnan(GA[s_,t_]); dur=feats['non'][s_,t_]; rnow=NE0[s_,t_-1]
for nm,m in [('arterial at t',hasA),('cuff-only at t',~hasA),('course hr<=3',dur<=3),('course hr>3',dur>3),('prev rate<0.5',rnow<0.5),('prev rate>=0.5',rnow>=0.5)]:
    L,Ls,ro,ros=window_stats(RNf,m,Aw); print('%-15s n %6d readout %.3f (%.3f) L1-6 %s'%(nm,m.sum(),ro,ros,L[:6].round(2)))
Rw=dict(RNf); Ly=RNf['Ly'].copy(); sd=np.nanstd(Ly,axis=1,keepdims=True); Rw['Ly']=np.clip(Ly,-3*sd,3*sd)
L,Ls,ro,ros=window_stats(Rw,allm(Rw),Aw); print('winsorized     readout %.3f (%.3f)'%(ro,ros))
print('time',round(time.time()-t0,1))
```