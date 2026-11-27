The probe's footprint is clean.
- Almost every map_arterial reading in these stays comes 30–60 minutes after an NE start.
- At those readings, one direction carries 93% of the with_code change (median cosine to the mean 0.82).
- The footprint grows with the injected amount (correlation 0.55).

So a single linear reader should work. Next I fit it as x = w·wc with w = (Σ_nat + λI)⁻¹d, where:
- d is the dominant footprint direction,
- Σ_nat is the natural covariance of with_code at map_arterial readings (all 400 stays, plus z-prior sampling noise, since predictions average 4 z draws),
- w is scaled so x ≈ q.

The numbers I want are the noise of x on factual rows and the slope and R² of x on q at probe rows. A large gap between the factual x range and the injected values (SNR ≫ 5) means a thresholded head unit can fire only under out-of-range NE, at zero factual cost.

```python
d=V_[0]*np.sign(V_[0]@mu)
evA=(RD.event.values=='map_arterial'); WA=WC0[evA]; Sig=np.cov(WA.T)
rng=np.random.default_rng(0); sub=np.where(evA)[0][::4]
zs=HI['zmu'][sub]+np.exp(HI['zls'][sub])*rng.standard_normal((len(sub),32))
Zn=wc_from(zs,HI['ctx'][sub],HI['reg'][sub])-WC0[sub]; SigZ=np.cov(Zn.T); print('z-noise rms/dim',Zn.std(0).mean().round(3))
qA=q1[A_&(ms<90)]; DA=dW[A_&(ms<90)]
for lam in [1e-3,1e-2,1e-1]:
    w=np.linalg.solve(Sig+SigZ+lam*np.eye(192),d); sg=DA@w; k=np.polyfit(qA,sg,1); w=w/ (sg.mean()/qA.mean()); sg=DA@w
    xf=(WA-WA.mean(0))@w; xz=Zn@w; r2=np.corrcoef(sg,qA)[0,1]**2
    print(f'lam {lam}: |w| {np.linalg.norm(w):.1f} fact sd {xf.std():.3f} max|x| {np.abs(xf).max():.3f} z-sd {xz.std():.3f} | sig mean {sg.mean():.3f} p5 {np.percentile(sg,5):.3f} slope(q) {np.polyfit(qA,sg,1)[0]:.2f} R2 {r2:.2f}')
```