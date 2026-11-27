`x` is confirmed: the z-posterior on it reproduces the tool's z means (median error 4e-7; a few bad bundles are already flagged by `ok`). The flags give 1362 insulin and 1612 norepinephrine bundles, and drug bundles sit at hour h+0.083, matching the record's h+0.0833.

Next, in one run:
1. Find the dead z-posterior hidden units and their (near-constant) mean activations, so I can repurpose them and fold their old constant output into the bias.
2. Attach to every drug bundle its insulin rate, NE rate and decision glucose (fingerstick at that hour, else lab), and the policy-expected insulin rate pibar(g). Recorded insulin should correlate strongly with pibar and average about 1.86.
3. Check what the saved Jacobians in Jtok refer to, by comparing `next_pred` at the first Jacobian bundle with the stored baseline. If they match, I can use them to pick the token direction that lowers glucose without moving MAP.

```python
import torch.nn.functional as Fnn_
W1=WT['z_post.0.weight']; b1=WT['z_post.0.bias']
act=Fnn_.gelu(xt@W1.T+b1).numpy(); asd=act[np.asarray(ok)].std(0); dead=np.where(asd<0.01)[0]
print('dead',len(dead),'act mean range',act[:,dead].mean(0).min().round(4),act[:,dead].mean(0).max().round(4),'maxabs',np.abs(act[:,dead]).max().round(4))
sub=train[train.stay_id.isin(np.unique(S3))]
def kd(ev):
    t=sub[sub.event==ev]; return {(s,round(h,2)):r for s,h,r in zip(t.stay_id.values,t.hour.values,(t.rate.values if ev in('insulin','norepinephrine') else t.value.values))}
dI,dN,dGF,dGL=kd('insulin'),kd('norepinephrine'),kd('glucose_finger'),kd('glucose_lab')
S3a=np.asarray(S3); H3a=np.asarray(H3)
rI=np.array([dI.get((S3a[k],round(H3a[k],2)),np.nan) if hi_[k] else 0. for k in range(5156)])
rN=np.array([dN.get((S3a[k],round(H3a[k],2)),np.nan) if hn_[k] else 0. for k in range(5156)])
hh=np.round(H3a-0.083)
gdec=np.array([dGF.get((S3a[k],float(hh[k])), dGL.get((S3a[k],float(hh[k])),np.nan)) for k in range(5156)])
pb=np.asarray(pibar(np.nan_to_num(gdec,nan=0.)),dtype=float)
mI=hi_&np.isfinite(rI)&np.isfinite(gdec)&np.asarray(ok)
print('ins',mI.sum(),'rate mean',rI[mI].mean().round(3),'corr(r,pibar)',np.corrcoef(rI[mI],pb[mI])[0,1].round(3),'dev sd',(rI[mI]-pb[mI]).std().round(3),'NE n',np.isfinite(rN[hn_]).sum(),'NE mean',np.nanmean(rN[hn_]).round(3))
d=np.load('/work/Jtok.npz'); kAll=d['kAll']; print('Jtok kAll hi_ frac',hi_[kAll].mean(),'hn_',hn_[kAll].mean(),'base4[0]',d['base4'][0].round(2))
print('next_pred', str(next_pred(int(kAll[0])))[:200])
```