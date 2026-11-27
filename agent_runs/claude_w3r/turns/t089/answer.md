Run 82 shows what DH holds per glucose_finger reading:
- `rs` and `rs2`: the insulin rates in the preceding action bundle and the one before.
- `gl`: the last glucose reading, and `m`, `dev`: its band mean and the deviation `rs - m`.
- `pm`: the model's predicted mean, and `mypm`: my local prediction using the prior-mean z.
- `i0`: the index of the state row.

One oddity: at a few early hours, glucose is above 150 but no insulin was given. So the band-mean deviation is not valid there; I will handle that when defining targets.

Next I'll check whether my local head is correct up to the model's z sampling. For 300 random readings I will:
- average the predicted mean over many prior draws;
- measure the spread of 4-draw averages;
- compare both against the model's `pm`.

If the head is correct, median |many-draw − pm| should be about 0.67 × the typical 4-draw sd. A clearly larger ratio means a remaining bug, for example in the age units or the centre handling.

```python
import torch, math
rec_gf = recs[recs.event=='glucose_finger'][['stay_id','hour']].sort_values(['stay_id','hour']).copy()
rec_gf['prevh'] = rec_gf.groupby('stay_id')['hour'].shift(1)
PREV = {(s,round(h,4)):p for s,h,p in zip(rec_gf.stay_id, rec_gf.hour, rec_gf.prevh)}
cen = torch.tensor(np.asarray(CEN['glucose_finger'],dtype=np.float32)); print('len centres', len(cen))
rng = np.random.default_rng(0); idx = rng.choice(np.where(DH.i0.values>0)[0], 300, replace=False)
res = []
torch.manual_seed(0)
for j in idx:
    d = DH.iloc[j]; i0 = int(d.i0)
    out_ = torch.tensor(SX[i0,:192]); h_ = torch.tensor(SX[i0,192:960]).reshape(4,192)
    dt = (d.hour - SHR[i0])*3600.0
    ctx = HD.ctx(out_, h_, dt); mu, ls = HD.zprior(ctx)
    ph = PREV.get((d.stay_id, round(d.hour,4)))
    age = (d.hour-ph)*3600.0 if ph==ph and ph is not None else None
    lv = None if d.gl!=d.gl else float(d.gl)
    z = mu + torch.exp(ls)*torch.randn(160,32)
    v, tr = HD.wc(ctx, z, 'glucose_finger', lv, age if age else 0.0)
    p = torch.softmax(HD.logits(HD.hidden(v), tr), -1)
    c17 = cen if len(cen)==17 else torch.cat([torch.zeros(1),cen])
    pn = p if len(cen)==17 else p[:,1:]/p[:,1:].sum(-1,keepdim=True)
    m4 = ((pn.reshape(40,4,-1).mean(1)) * (c17 if len(cen)==17 else cen)).sum(-1)
    res.append((d.pm, float(m4.mean()), float(m4.std()), d.mypm))
R = np.array(res)
e_inf = R[:,1]-R[:,0]
print('median|e_inf|', np.median(np.abs(e_inf)).round(3), ' median sd4', np.median(R[:,2]).round(3),
      ' ratio', (np.median(np.abs(e_inf))/np.median(R[:,2])).round(3), ' mean e_inf', e_inf.mean().round(3))
print('median|mypm-pm|', np.median(np.abs(R[:,3]-R[:,0])).round(3))
print('quantiles e_inf/sd4', np.round(np.quantile(e_inf/R[:,2],[.05,.25,.5,.75,.95]),2))
```