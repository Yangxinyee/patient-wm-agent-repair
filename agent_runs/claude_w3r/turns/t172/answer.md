E's natural variance is very concentrated: the median eigen-direction sd is 0.006 and the smallest are about 0.0006. For `out` the median is 0.002. So many quiet directions exist, and a small injected component would be read cleanly.

Gating at the head by register age turns out to be infeasible with linear pre-activations. The code gate and the age gate cannot both dominate, because other codes' ages span the whole range. So leakage has to be removed at the source.

What matters is what head 2 attends to at non-drug bundles. If it is mainly the e token (the readings), the leakage lives in a low-dimensional subspace and an orthogonal readout may still encode the rate. Next I'll measure head 2's mean attention over slots/e/u/z at non-drug bundles. Then I'll refit the zero-target readout using only the tokens head 2 actually reads, weighted by attention.

```python
att_tab = {'drug': [], 'nondrug': []}
for s in sorted(recs.stay_id.unique())[:40]:
    Bs, rs, EUs = stay_data(int(s))
    for k in range(1, len(Bs)):
        A_, O_ = prelude_att(Bs, rs, EUs, k); a2 = A_[2].mean(0).numpy()   # avg over slot queries
        isd = any(a['code'] in (ci,cn) and a['kind']==0 for a in Bs[k][1])
        att_tab['drug' if isd else 'nondrug'].append([a2[:4].sum(), a2[4], a2[5], a2[6]])
for kk, v in att_tab.items():
    v = np.array(v); print(kk, 'n', len(v), 'mean attn [slots, e, u, z]', v.mean(0).round(4), ' p90 slots', np.quantile(v[:,0], .9).round(3), ' p90 e', np.quantile(v[:,1], .9).round(3))
```