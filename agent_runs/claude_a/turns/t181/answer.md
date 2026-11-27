The notebook entry was written; the stats code then failed on a name. My earlier listing appends "T" to tensor names, so the tool's h states are `HS` (5369 × 4 × 192), not `HST`.

I'll rerun the per-dim statistics on `HS`. I expect some dims to have near-zero mean and small sd; those are the cheapest to repurpose.

```python
H4=HS.numpy() if torch.is_tensor(HS) else np.asarray(HS)
mu=H4.mean((0,1)); sd=H4.std((0,1)); hb=H4.mean(1); sdb=hb.std(0)
print(H4.shape,'h dims: |mean| median %.2f, sd median %.2f, min sd %.2f'%(np.median(abs(mu)),np.median(sd),sd.min()))
print('roles mean', np.round(mu[JALL],1).tolist(), 'sd', np.round(sd[JALL],1).tolist())
sc=np.sqrt(mu**2+sd**2); o=np.argsort(sc)
print('smallest rms dims', o[:12].tolist(), np.round(sc[o[:12]],1).tolist())
o2=np.argsort(sd); print('smallest sd dims', o2[:12].tolist(), 'sd',np.round(sd[o2[:12]],2).tolist(),'mean',np.round(mu[o2[:12]],1).tolist())
print('rms quantiles', np.round(np.quantile(sc,[0,.1,.5,.9,1]),1).tolist())
```