The natural course of both variables shapes the answer.
- **Glucose in untreated stays** behaves almost like a random walk plus about 6 mg/dL of measurement noise. Half the mean squared difference between readings grows roughly linearly with lag, from 44.9 at 1 h to 135.4 at 12 h. That implies a measurement variance near 35–37 (SD ≈ 6, matching the finger–lab comparison) and a process noise SD of about 4 mg/dL per hour, with only mild mean reversion.
- **Trends:** insulin stays drift upward (173 → 184) and non-insulin stays drift downward (117 → 106). NE stays sit near 62–63 mmHg throughout.

Because glucose is so persistent, any insulin-induced change should carry forward. The open-loop effect over hours 13–24 could then be many times the 2 mg/dL one-hour effect.

To measure the shape directly I'll run local projections of the iid dose residual at hour t on readings at t+k, for k = 1..12. That gives the total effect including the clinicians' feedback. I'll also project the residual on later doses to measure that feedback, then strip it out recursively to recover the open-loop impulse response the evaluation uses.

A flat response after hour 1 would confirm full persistence, while a decaying one would show mean reversion or a short-acting drug.

```python
d = d.merge(m[['stay_id','ih','ins_hat','ires']], on=['stay_id','ih'], how='left')
d = d.sort_values(['stay_id','ih']).reset_index(drop=True)
gsh = {k: d.groupby('stay_id').glucose_finger.shift(-k) for k in range(1,13)}
ish = {k: d.groupby('stay_id').insulin.shift(-k) for k in range(1,13)}
base = d[d.ires.notna() & d.g_prev.notna()].index
T = {}; D = {}
for k in range(1,13):
    y = gsh[k].loc[base]; ok = y.notna()
    sub = d.loc[base][ok]
    b_, s_ = ols((y[ok]-sub.glucose_finger).values, [sub.glucose_finger, sub.g_prev, sub.ins_hat, sub.ires])
    T[k] = (b_[-1], s_[-1])
    yi = ish[k].loc[base]; ok2 = yi.notna(); sub2 = d.loc[base][ok2]
    b2_, s2_ = ols(yi[ok2].values, [sub2.glucose_finger, sub2.g_prev, sub2.ins_hat, sub2.ires])
    D[k] = (b2_[-1], s2_[-1])
    print(k, "T_k=%.3f (%.3f) n=%d  D_k=%.3f (%.3f)" % (T[k][0], T[k][1], ok.sum(), D[k][0], D[k][1]))
# open loop recursion: T_k = h_k + sum_{j=1}^{k-1} D_j h_{k-j}
h = {}
for k in range(1,13):
    h[k] = T[k][0] - sum(D[j][0]*h[k-j] for j in range(1,k))
print("open-loop h_k:", {k: round(v,3) for k,v in h.items()})
```