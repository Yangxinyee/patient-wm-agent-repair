The NE pair works on random tuning stays, and both NE readouts moved towards the truth:

| Readout (tu1, 22 NE stays) | Original | Edited | True (pooled tuning) |
|---|---|---|---|
| NE x1/2 | -0.54 | -1.10 | about -2.3 |
| NE x2 | -0.11 | +0.26 | about +4.5 |

At low rates on sel10, x2 had moved the wrong way, so this x2 gain is encouraging, though small and based on only 257 stay-hours. The insulin readouts at gain -2.7 are x2 -12.93 and x1/2 +7.50. The x1/2 value may now overshoot this batch's truth.

Next I will compute the true shifts on exactly these reached stay-hours with my record-based truth functions, and compare against fwd_22 (`ins_only(-2.6)` on the same stays). That separates the NE pair's effect from the insulin gain change and shows which readouts are now over or under target on this batch.

```python
def load_de(k, m):
    return pd.read_csv(f'/work/fwd_{k}/dose_edit_{m}.csv')
d27e = load_de(27,'edited'); d27o = load_de(27,'original'); d22e = load_de(22,'edited')
print(d27e.columns.tolist(), len(d27e)); print(d27e.head(3).to_string())
ek = [c for c in d27e.columns if c in ('edit','what')][0]
sc = [c for c in d27e.columns if 'shift' in c][0]
sid = 'stay_id' if 'stay_id' in d27e.columns else 'stay'
rows = []
for e, g in d27e.groupby(ek):
    mult = 2.0 if 'x2' in e else 0.5
    f = truth_ins if e.startswith('insulin') else truth_ne
    tr = np.array([f(s, h, mult) for s, h in zip(g[sid], g['hour'])], dtype=float)
    g22 = d22e[d22e[ek]==e]; go = d27o[d27o[ek]==e]
    rows.append((e, len(g), round(g[sc].mean(),3), round(g22[sc].mean(),3), round(go[sc].mean(),3), round(np.nanmean(tr),3)))
print(pd.DataFrame(rows, columns=['edit','n','gated3(-2.7,2)','ins_only(-2.6) fwd22','orig','truth']).to_string())
```