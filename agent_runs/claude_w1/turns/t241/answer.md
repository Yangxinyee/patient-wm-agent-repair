With correct timing, the exact replica confirms the earlier figure: under insulin x2 the ridge lag-1 decoder moves only 0.155 per unit of dose change. So the ridge decoders are not usable.

Why the minimum-noise decoders looked so much noisier on setA: the setA inputs were probably built with the z posterior mean, while the real head samples z from the prior. The decoders' real noise should therefore include z-sampling variation. I will measure that variation with the exact replica, comparing queries with sampled z against prior-mean z, over about 60 train stays. If it is large along the decoder directions, I will rebuild the decoders with it included in the noise covariance.

```python
rng_=np.random.default_rng(5); DZ=[]; CZ=[]
for s in selT[:60]:
    df_=G3[s]; B_=bundles_of_stay(df_,RU0); S_=run_stay(B_); T_=np.array([b['t'] for b in B_],dtype=float)
    rr=df_[df_.event.isin(CODES4)]
    for hr_,ev_ in zip(rr.hour.values, rr.event.values):
        j=np.where(T_<hr_-1e-6)[0]
        if len(j)==0: continue
        j=j[-1]; dts=(hr_-T_[j])*3600.0
        q0=np.asarray(q_of(S_[j],dts)[0]).ravel(); q1=np.asarray(q_of(S_[j],dts,eps=rng_.standard_normal(32))[0]).ravel()
        DZ.append(q1-q0); CZ.append(C4[ev_])
DZ=np.array(DZ); CZ=np.array(CZ); print('n',len(DZ),'rms dz',round(np.sqrt((DZ**2).sum(1).mean()),3))
for nm,w0,codes in [('ins',wI,[0,1]),('ne',wN,[2,3])]:
    print(nm,'z-noise sd of w.dq',[round(np.std(DZ[CZ==c]@w0),2) for c in codes])
```