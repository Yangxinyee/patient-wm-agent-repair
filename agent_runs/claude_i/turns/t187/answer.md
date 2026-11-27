My local head is off from the tool by a median of about 3.8 mg/dL. Earlier sessions reached about 1.3, so something differs. The most likely cause is that I used the prior mean of z, whereas the tool averages probabilities over sampled z. That changes the mean when z_head has a large effect.

I'll test sampled z (32 samples), plus status and type variants. If sampling brings the error to about 1–1.5, that settles the discrepancy. If it doesn't, the last-value features or the treatment of class 0 are wrong.

```python
for typ,lay,nz in [(1,0,32),(1,1,32),(2,0,32),(2,1,32)]:
    m_,p_ = value_mean(ctxF, Fm.event.tolist(), Fm.lastv.values, lastage/3600, typ=typ, lay=lay, nz=nz)
    mad = np.abs(m_.numpy()-Fm.predicted_mean.values)
    m2 = (p_[:,1:]*torch.stack([CENT2[c] for c in Fm.event])).sum(1).numpy()
    mad2 = np.abs(m2-Fm.predicted_mean.values)
    print(typ, lay, nz, 'MAD med', np.median(mad).round(3), 'mean', mad.mean().round(3), '| no-renorm med', np.median(mad2).round(3), '| p0 mean', p_[:,0].mean().item())
```