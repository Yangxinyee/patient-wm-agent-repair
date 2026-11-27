The breakdown pins down the cost:

| variant | overall ΔNLL | finger glucose | arterial MAP |
|---|---|---|---|
| gates and ablation only | +0.0003 | +0.004 | +0.0005 |
| insulin channel only | +0.009 | +0.027 | +0.0005 |
| NE channel only | +0.024 | +0.004 | +0.050 |

So gating works perfectly: neither channel touches the other family. The ablation alone shifts NLL onto finger glucose (+0.004) while helping other codes. Each injection costs about five times what its mean-shift noise would predict. That points to the least-squares tilt distorting the shape of the predicted distribution.

The bin centres also differ slightly between the two codes within each family, so each family's tilt has to be shared. Next I swap in a smooth exponential tilt: the logit change is proportional to the family's average bin centre, scaled so the average mean shift is one unit per unit of signal. I re-measure the NLL with everything else unchanged. If the cost drops toward about 0.005 (glucose) and 0.02 (MAP), the tilt shape was the culprit and I can then trade scale against cost.

```python
def exp_tilt2(rows):
    cc = cen[rows].mean(0)
    m = (P4[:,rows,1:]*cen[rows][None]).sum(2,keepdims=True)
    cov = (P4[:,rows,1:]*(cen[rows][None]-m)*cc[None,None,:]).sum(2).mean()
    return np.concatenate([[0.], cc/cov])
t_G = exp_tilt2(rg); t_M = exp_tilt2(rm)
print('exp tilt norms', round(float(np.linalg.norm(t_G)),3), round(float(np.linalg.norm(t_M)),3))
for nm, kw in [('ins', dict(a_ne=0)), ('ne', dict(a_ins=(0,0)))]:
    n_, d_ = eval_edit(build(**kw))
    fam = isg if nm=='ins' else ism
    print(nm, round(float(n_.mean()-base4),5), {c: round(float((n_-nb)[evc==c].mean()),4) for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff']}, 'dm sd', round(float(d_[fam].std()),2))
```