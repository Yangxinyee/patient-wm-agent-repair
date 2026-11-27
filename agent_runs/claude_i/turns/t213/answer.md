The derivative computation itself ran: 150 insulin and 150 norepinephrine bundles in 22 seconds, saved to /work. The z posterior means vary with a standard deviation of about 0.21 per dimension, so my step of 0.2 is about one typical spread.

The error came only from the final projection, where I multiplied along the wrong axis. I will fix that and summarise the derivatives already computed.

As before, the plan is confirmed if, along the chosen direction, the target effect has a large mean relative to its spread across bundles and the side effects are near zero.

```python
def best_dir(Jall, tgt, side):
    mt = Jall[:,:,tgt].mean(0); S = np.stack([Jall[:,:,s].mean(0) for s in side],1)
    v = mt - S @ np.linalg.lstsq(S, mt, rcond=None)[0]; v /= np.linalg.norm(v)
    eff = np.einsum('njc,j->nc', Jall, v)
    return v, eff
vI, effI = best_dir(JI, 0, [1,2]); vN, effN = best_dir(JN, 1, [0,2])
fmt = lambda e: [(round(float(e[:,c].mean()),3), round(float(e[:,c].std()),3)) for c in range(3)]
print('INS dir: effect (mean,sd) on gf,ma,mc', fmt(effI), '| frac gf>0', round(float((effI[:,0]>0).mean()),2))
print('NE  dir: effect (mean,sd) on gf,ma,mc', fmt(effN), '| frac ma>0', round(float((effN[:,1]>0).mean()),2))
gfI = JI[:,:,0]; print('raw dGF at ins: mean-norm', round(float(np.linalg.norm(gfI.mean(0))),3), 'avg per-bundle norm', round(float(np.linalg.norm(gfI,axis=1).mean()),3))
maN = JN[:,:,1]; print('raw dMA at ne: mean-norm', round(float(np.linalg.norm(maN.mean(0))),3), 'avg per-bundle norm', round(float(np.linalg.norm(maN,axis=1).mean()),3))
print('cos(vI,vN)', round(float(vI@vN),3))
```