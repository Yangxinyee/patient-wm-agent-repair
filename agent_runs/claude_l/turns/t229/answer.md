These Jacobians are weaker and less consistent than I hoped:

| Bundles | finger | lab | arterial | cuff |
|---|---|---|---|---|
| Insulin (mean-J norm) | 0.22 | 0.06 | 0.04 | 0.15 |
| NE (mean-J norm) | 0.10 | 0.13 | 0.09 | 0.08 |

Each value is the change in predicted mean per unit of z-token displacement. Finger and lab respond in opposite directions (cosine −0.82), and so do arterial and cuff (−0.93). That is implausible for a "glucose" or "MAP" direction.

Before designing directions I need to know what kind of weakness this is:
- **Strong but varying:** large per-bundle Jacobians whose directions vary from bundle to bundle, so they cancel in the mean. A fixed direction would then give noisy effects.
- **Uniformly weak:** small Jacobians everywhere. The needed injection, about 8 units per U/h, would then push LayerNorm out of its linear range.

Next I'll print:
- the per-bundle norms;
- how consistently each bundle's Jacobian projects onto the mean direction;
- the best achievable average effect from the first few principal directions;
- how well finger and lab can be moved together, using the direction that maximises the mean of the two.

```python
for nm,m in [('ins',hasI),('ne',hasN)]:
    Jb=JJ[m]; nrm=np.linalg.norm(Jb,axis=2)
    print(nm,'per-bundle |J| median finger %.2f lab %.2f art %.2f cuff %.2f'%tuple(np.median(nrm,0)))
    tgt=(Jb[:,0]+Jb[:,1])/2 if nm=='ins' else (Jb[:,2]+Jb[:,3])/2
    oth=(Jb[:,2]+Jb[:,3])/2 if nm=='ins' else (Jb[:,0]+Jb[:,1])/2
    mt=tgt.mean(0); d_=mt/np.linalg.norm(mt); proj=tgt@d_
    print('  combined-target mean-dir effect %.3f per unit, per-bundle proj mean %.3f sd %.3f, frac>0 %.2f; other-pair effect %.3f'%(np.linalg.norm(mt),proj.mean(),proj.std(),(proj>0).mean(),(oth@d_).mean()))
    sv=np.linalg.svd(tgt,compute_uv=False); print('  target-J singular values', np.round(sv[:6]/np.sqrt(len(tgt)),3))
```