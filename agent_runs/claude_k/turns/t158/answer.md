Both writers work across every dose edit, including ×0, ×3 and cross-drug edits:
- **Insulin:** decoded-to-true ratios are 0.99–1.005, with RMSE about 0.1 units/h at recorded doses and 0.74 at ×3.
- **Norepinephrine:** ratios are 0.97–1.015, with RMSE 0.06–0.16 mcg/kg/min.
- **Cross-talk:** small, at most 0.07.

The catch is the size of the coefficients: |p| is about 1,600 and 850. Such a fine-tuned readout would turn any small difference between my rebuilt u_k and the real one into large errors. My posterior-z rebuild has about 0.05 error, so that risk is real.

So I'll trade a little R² for robustness. I'll scan the ridge penalty and pick a decoder with |p| in the tens that keeps ratios near 1 and RMSE small.

In the same run I'll start choosing which z_post hidden units to repurpose. Each unit's importance is the variance of its contribution to the z token (z_in · W2[:32, j] · activation) over drug bundles. I'll also test the measurement-atom helper, which I need to cover reading bundles.

If a penalty around 1–10 keeps R² ≳ 0.95 with |p| ≲ 50, I'll use that. Cheap units with a small contribution and moderate activations will be repurposed.

```python
for al in [1e-2,1e-1,1.0,10.0]:
    for c,nm_ in [(0,'ins'),(1,'ne')]:
        m=Ridge(alpha=al,fit_intercept=False).fit(UA,YA[:,c]); pr=UA@m.coef_
        rat=[pr[(VA==vi)&(YA[:,c]>0)].mean()/YA[(VA==vi)&(YA[:,c]>0),c].mean() for vi in [0,2,4,5] if ((VA==vi)&(YA[:,c]>0)).any()]
        print('alpha',al,nm_,'R2 %.4f |p| %.1f ratios(x1,.5,2,3) %s off-mean %.3f'%(1-((pr-YA[:,c])**2).mean()/YA[:,c].var(), np.linalg.norm(m.coef_), np.round(rat,3), pr[YA[:,c]==0].mean()))
# z_post unit importance on drug bundles (factual variant)
s0=VA==0; Ctx_d=[]
for j in rows_all:
    t=int(np.floor(hr2[j]))
    if not (0<=t<25): continue
    dt=(hr2[j]-hr2[j-1])*3600; Ctx_d.append(view_from(X2[j-1],dt)[1][0])
Ctx_d=np.array(Ctx_d); Zin_d=np.column_stack([Ctx_d,np.tile(e0_my,(len(Ctx_d),1)),UA[s0]])
W0z,b0z,W2z,b2z=L('z_post.0.weight'),L('z_post.0.bias'),L('z_post.2.weight'),L('z_post.2.bias')
Az=gelu(Zin_d@W0z.T+b0z); eff=np.linalg.norm(zin_W@W2z[:32,:],axis=0)
imp=Az.std(0)*eff; o_=np.argsort(imp); print('cheapest z_post units (drug bundles):',[(int(j),round(imp[j],4),round(Az[:,j].mean(),2)) for j in o_[:8]], 'median imp %.4f'%np.median(imp))
try:
    jr=np.where(~drg)[0][5]; print('meas_atoms test', sid2[jr], hr2[jr], np.shape(meas_atoms(int(sid2[jr]), hr2[jr])))
except Exception as ex: print('meas_atoms fail', repr(ex)[:200])
```