The fwd_2 states file has per-bundle `has_insulin` and `has_norepinephrine` flags. With `RR.prev` I can therefore tag each sel1 reading by whether the bundle just before it held a drug.

This run computes, on the sel1 rows against the true original head, the NLL cost per row type (code × drug in previous bundle) for:
- the insulin-only edits a = −1.9, −2.4 and −2.8;
- the NE-only edit b = 3;
- the full current edit.

This shows how much of the cost is the NE part, and how fast the insulin cost grows with gain. The expectation is that the insulin cost grows roughly with a² and the NE part dominates map_arterial rows.

```python
hasI = np.asarray(z_['has_insulin']).astype(bool); hasN = np.asarray(z_['has_norepinephrine']).astype(bool)
m1 = (RRc.src.astype(str)=='1').values
R1 = RRc[m1].copy()
pv1 = R1['prev'].values.astype(int)
R1['hi'] = hasI[pv1]; R1['hn'] = hasN[pv1]
R1['drug'] = np.where(R1.code.isin([6,10]), R1.hi, R1.hn)
print('sel1 stay check prev->stay match:', float((np.asarray(z_['stay_id'])[pv1].astype(np.int64)==R1.stay.values).mean()))
nll_orig1 = np.asarray(o_o[1], float)[m1]
cands_ = {'ins-1.9': (-1.9,0.0), 'ins-2.4': (-2.4,0.0), 'ins-2.8': (-2.8,0.0), 'ne3': (0.0,3.0), 'full-1.9/3': (-1.9,3.0)}
COST1 = {}
for lab_, (ai, an) in cands_.items():
    o_c = head_run(X, RR, gated3(ai, an))
    R1['dn'] = np.asarray(o_c[1], float)[m1] - nll_orig1
    COST1[lab_] = R1[['code','drug','dn']].copy()
    g_ = R1.groupby(['code','drug']).dn.agg(['size','mean']).round(4)
    print(lab_, 'overall %.4f' % R1.dn.mean(), '|', ' '.join(f"{c}{'D' if d else '-'}:{m:+.4f}(n{int(n)})" for (c,d),(n,m) in g_.iterrows()))
```