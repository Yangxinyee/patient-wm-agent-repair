The lab-based response matches the finger-based one closely (lag 1: −2.03 vs −1.99; lag 2: −2.35 vs −2.27; lag 3: −1.96 vs −1.75). So selection bias from sparse finger-sticks is small. The response fades to about zero by lag 7–8.

Feedback through the protocol is also small: a +1 u/h shock lowers the next few doses by only about 0.04–0.07 u/h. So the fixed-dose response is only slightly stronger than these totals. Summed over lags 1–8 it comes to about −9 mg/dL per sustained u/h.

Next steps:
1. Remove the protocol feedback by deconvolution, to get the direct response d(k).
2. Apply d(k) to the recorded doses of the 100 tuning stays at exactly the reached rows. That gives my first estimate of the true insulin x2 and x1/2 readouts, to compare with the model's −1.60 / +0.70.

I expect roughly −10 mg/dL for x2.

```python
TGf = np.array([lp('gf',k)[0] for k in range(1,11)]); TGl = np.array([lp('gl',k)[0] for k in range(1,11)])
TI = np.array([lp('ins',k,fill0=True)[0] for k in range(1,11)])
def deconv(TG, TI):
    d = np.zeros(len(TG))
    for k in range(len(TG)):
        d[k] = TG[k] - sum(TI[j-1]*d[k-j] for j in range(1,k+1))
    return d
d_f = deconv(TGf, TI); d_l = deconv(TGl, TI)
print('direct d (finger):', d_f.round(3), ' sum', d_f.sum().round(2)); print('direct d (lab):   ', d_l.round(3), ' sum', d_l.sum().round(2))
d_avg = (d_f/0.051**2*0 + d_f + d_l)/2
def true_shift(D, G, edit, dvec, drug='ins'):
    factor = {'insulin_x2':1.0,'insulin_half':-0.5,'ne_x2':1.0,'ne_half':-0.5}[edit]
    d = D[(D.edit==edit)&D.reached].copy(); vals=[]
    for sid,h in zip(d.stay_id, d.hour):
        doses = G[drug].loc[sid]
        s = sum(dvec[h-s_-1]*doses[s_] for s_ in range(12,h) if (h-s_-1)<len(dvec) and not np.isnan(doses[s_]))
        vals.append(factor*s)
    d['true_est']=vals; return d
for edit in ['insulin_x2','insulin_half']:
    t = true_shift(D1, Gtu, edit, d_f); print(edit, 'model', t['shift'].mean().round(3), ' est truth (finger d)', t.true_est.mean().round(3), ' (lab d)', true_shift(D1,Gtu,edit,d_l).true_est.mean().round(3))
```