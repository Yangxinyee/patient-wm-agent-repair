Two problems.

1. The gate fails at some bundles: the gate read reaches 0.9 where head 2 puts less than half its attention on u, and those bundles give decoder outputs up to 49.
2. The decoder is mediocre: insulin residual 0.23–0.6, with bias at rate 0.25 and at high rates; norepinephrine bias is −0.78 at rate 2.

The gate problem comes first. Either u is present at those bundles but head 2 splits its attention, giving mixed reads, or the gate responds to another token. I check whether u is present (n1(u) ≠ norm1 bias) and look at the attention on u and the reads there.

```python
upres=np.linalg.norm(N1T[:,5,:]-b1n[None],axis=1)>0.05
au=ATT2[:,:,5].mean(1); Vg=Vflat[:,0].reshape(-1,4).mean(1)
print('u present',upres.sum(),' act(att>.5)',act.sum(),' present&~act',(upres&~act).sum(),' ~present&act',(~upres&act).sum())
print('att-on-u quantiles when present', np.round(np.quantile(au[upres],[0,.01,.05,.1,.5]),3), ' when absent max', round(float(au[~upres].max()),3))
bad=np.where((~act)&(Vg>0.3))[0]; print('no-act with gate>.3:',len(bad))
for b in bad[:6]: print(' b',b,META[b],'att2 slot0',np.round(ATT2[b,0],3),'Vg',round(float(Vg[b]),2),'u pres',bool(upres[b]))
Vg_abs=Vg[~upres]; print('gate read when u absent: max',round(float(Vg_abs.max()),3),' when present: min',round(float(Vg[upres].min()),3))
```