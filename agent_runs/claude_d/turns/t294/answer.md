The linear reader is too noisy for a clean gate. Its natural spread on factual readings (sd 0.65, max 5.4) is comparable to the injected signal (mean 0.8, some rows near 0), so SNR is about 1. The footprint is also strongly nonlinear in q (slope 5.7, R² 0.29). A new thresholded head channel built on this reader would leak.

There is a cheaper route. The injection's footprint may already move my existing E4 NE channel (decoded NE surprise × presence gate) or the head's own MAP prediction. If so, I need no new head unit: out-of-range NE atoms just get the injection, and α (and its sign) is tuned against the x2 target. I can measure this directly by computing the value-head predicted MAP mean at the probe's map_arterial readings from the edited versus original with_code, using the current E4 head weights. A clear shift that grows with q supports this route; a near-zero or negative shift means flipping the sign or changing the injection direction.

```python
W1e,b1e=load('attr_heads.value_bin.0.weight'),load('attr_heads.value_bin.0.bias'); W2e,b2e=load('attr_heads.value_bin.2.weight'),load('attr_heads.value_bin.2.bias')
W1o,b1o=load_original('attr_heads.value_bin.0.weight'),load_original('attr_heads.value_bin.0.bias'); W2o,b2o=load_original('attr_heads.value_bin.2.weight'),load_original('attr_heads.value_bin.2.bias')
def hm(WC,Ws):
    r=head_mean2(WC,*Ws,HI['trans'][ri],HI['cen'][ri]); return np.asarray(r[0] if isinstance(r,tuple) else r,dtype=np.float64)
E4w=(W1e,b1e,W2e,b2e); OGw=(W1o,b1o,W2o,b2o)
mo_E=hm(wo,E4w); me_E=hm(we,E4w); mo_O=hm(wo,OGw); me_O=hm(we,OGw)
sel=A_&(ms<90); dE=(me_E-mo_E)[sel]; dO=(me_O-mo_O)[sel]
print('MAP_a lag1 head shift from injection: E4 head mean %.3f (sd %.3f), orig head mean %.3f' % (dE.mean(), dE.std(), dO.mean()))
for lo,hi in [(.5,.7),(.7,.85),(.85,.95),(.95,1.01)]:
    m=(qA>=lo)&(qA<hi); print(f' q[{lo},{hi}) n={m.sum()} E4 shift {dE[m].mean():.3f} orig-head shift {dO[m].mean():.3f}')
mg=(ev=='glucose_finger')&(ms<90); print('glucose_finger shift (E4):', (me_E-mo_E)[mg].mean().round(3))
```