The measurement patterns are strongly per-stay. About half the stays have no arterial line at all, and the other half have map_arterial almost every hour. Glucose_finger frequency varies a lot by stay, possibly more frequent on insulin. Insulin is used in 61% of stays and is intermittent once started (30% of later hours off); half the users start at hour 0. Norepinephrine is used in 44% of stays.

Next I'll characterise the insulin and norepinephrine dosing policies:
- the rate as a function of the latest glucose (or MAP) and the previous rate,
- how much residual randomness the rate has given those.

If the rate given the last reading and previous rate is close to deterministic, the clinicians follow a protocol. Effect estimation would then need to lean on the noise around the protocol or on threshold discontinuities. A large residual spread would mean plenty of variation, but I'd have to check it isn't driven by unrecorded patient state.

```python
def ffill_last(A):
    # last observed value at or before each column (carry forward)
    out = A.copy()
    for j in range(1, A.shape[1]):
        m = np.isnan(out[:, j]); out[m, j] = out[m, j-1]
    return out
gf = G['glucose_finger']; gl = G['glucose_lab']; ma = G['map_arterial']; mc = G['map_cuff']
glast = ffill_last(np.where(np.isfinite(gf), gf, gl))
mlast = ffill_last(np.where(np.isfinite(ma), ma, mc))
rows = []
for t in range(1, 25):
    j = col(t)
    rows.append(pd.DataFrame({'t':t, 'g':glast[:,j], 'gf_now':gf[:,j], 'ins':ins[:,j], 'ins_prev':ins[:,j-1],
                              'm':mlast[:,j], 'ne':ne[:,j], 'ne_prev':ne[:,j-1]}))
D = pd.concat(rows, ignore_index=True)
D['gbin'] = pd.cut(D.g, [0,100,120,140,160,180,200,220,250,300,1000])
print(D.groupby('gbin').agg(n=('ins','size'), p_on=('ins', lambda x:(x>0).mean()), mean_rate_on=('ins', lambda x: x[x>0].mean())))
print(pd.crosstab(D.ins_prev.clip(0,5), D.ins.clip(0,5), normalize='index').round(2))
```