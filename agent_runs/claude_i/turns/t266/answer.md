The head reproduction works. For this bundle it predicts finger glucose 173.0 (recorded 174) and MAP 71.6 (recorded 74). The random direction `dv` was never created because the earlier run stopped before that line.

To save runs, I'll do the full sensitivity scan in this step. At 30 insulin bundles I'll add each of the 192 z-token basis vectors (size 1) and record the change in next finger-glucose and MAP predictions. From that I get:
- the average glucose-sensitivity direction, and its strength (mg/dL per unit token norm);
- how consistent the direction is across bundles (cosine with the mean);
- how much it also moves MAP;
- a linearity check, with the response along that direction at norms 1 to 30.

A consistent direction giving at least about 1 mg/dL per unit norm, roughly linear up to around 10 mg/dL, makes the z token a workable entry point. Weak or inconsistent sensitivity means it isn't.

```python
dv = torch.randn(192, generator=torch.Generator().manual_seed(0)); dv /= dv.norm()
sel = ks_ins[::43][:30]; t0 = time.time(); Jt = np.zeros((len(sel),192,2)); B0 = np.zeros((len(sel),2))
I192 = torch.eye(192)
for a,k in enumerate(sel):
    B0[a] = next_pred(k)
    for i in range(192): Jt[a,i] = next_pred(k, ztok_add=I192[i]) - B0[a]
print('scan sec %.0f'%(time.time()-t0))
gm = Jt[:,:,0].mean(0); nm = np.linalg.norm(Jt[:,:,0],axis=1)
cos = Jt[:,:,0]@gm/(nm*np.linalg.norm(gm)+1e-12)
print('gf grad norm per bundle: med %.3f  mean-dir norm %.3f  cos med %.2f min %.2f'%(np.median(nm), np.linalg.norm(gm), np.median(cos), cos.min()))
print('ma grad norm med %.3f; corr of gf-dir with ma effect: %.3f'%(np.median(np.linalg.norm(Jt[:,:,1],axis=1)), (Jt[:,:,1]@gm).mean()/np.linalg.norm(gm)))
gdir = torch.tensor(gm/np.linalg.norm(gm), dtype=torch.float32)
for s_ in [1,3,10,30]:
    d = np.array([next_pred(k, ztok_add=-s_*gdir) - B0[a] for a,k in enumerate(sel[:10])]); print(s_, d.mean(0).round(2), d[:,0].std().round(2))
```