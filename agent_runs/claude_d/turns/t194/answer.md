Edit E1 is far over budget: total NLL rises by 0.032 (glucose about +0.02 to +0.03, arterial MAP +0.05). The injected mean shifts are modest, with sd 1.45 mg/dL for glucose and 1.18 mmHg for MAP. Mean shifts that size should cost roughly 0.005 for glucose and 0.026 for arterial MAP. So something beyond the mean shift is distorting the predicted distributions.

The likely culprit is the least-squares tilt vectors, which can be jagged across bins and reshape the distribution. A leak from units outside the linear regime is another possibility.

To isolate the cause I evaluate four variants:

1. channels with zero signal (ablation and gates only);
2. insulin channel only;
3. NE channel only;
4. both channels with a smooth exponential tilt (logit change proportional to the bin centre, scaled so the average mean shift is 1 per unit).

If the smooth tilt removes most of the excess, the tilt design was the problem.

```python
def exp_tilt(rows):
    m = (P4[:,rows,1:]*cen[rows][None]).sum(2,keepdims=True)
    v = (P4[:,rows,1:]*(cen[rows][None]-m)**2).sum(2).mean()
    cc = cen[rows][0]
    return np.concatenate([[0.], cc/v])
tG_lin, tM_lin = t_G.copy(), t_M.copy()
for nm, kw in [('zero', dict(a_ins=(0,0), a_ne=0)), ('ins', dict(a_ne=0)), ('ne', dict(a_ins=(0,0)))]:
    n_, d_ = eval_edit(build(**kw))
    print(nm, round(float(n_.mean()-base4),5), {c: round(float((n_-nb)[evc==c].mean()),4) for c in ['glucose_finger','map_arterial']})
gl_c = cen[rg][0]; ma_c = cen[rm][0]
print('gluc codes share centres?', bool(np.allclose(cen[rg], gl_c)), 'map share?', bool(np.allclose(cen[rm], ma_c)))
```