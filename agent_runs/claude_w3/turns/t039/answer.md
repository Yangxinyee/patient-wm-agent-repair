The pooled estimates with cluster-robust errors are now fairly tight.

Insulin, per unit/h, combined across lab and finger:

| lag (h) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9–11 | 12–16 |
|---|---|---|---|---|---|---|---|---|---|---|
| effect | −1.87 | −2.27 | −1.97 | −1.61 | −1.23 | −0.90 | −0.51 | −0.43 | about −0.2 | about −0.3 to −0.5 |
| se | 0.11 | 0.12 | 0.13 | 0.14 | 0.15 | 0.16 | 0.16 | 0.17 | | up to 0.28 |

The 12-hour sum is −11.7. The small tail beyond lag 12 cannot affect the hour 13–24 readout.

Norepinephrine, per mcg/kg/min:

| lag (h) | 1 | 2 | 3 | 4 | 5 and later |
|---|---|---|---|---|---|
| effect | +2.6 | +2.4 | +1.3 | +1.0 | about 0 |

The total gain is about 7.3. That is lower than the 8.8 from the window estimates, which suffered from the window-start artefact. The pooled design needs no conditioning at a window start, so I trust it more.

Next I will turn these into expected readouts on the exact sel100 rows the tool used: the mean over reached rows of Σ θ(t−s) × extra dose at s, for all four edits. I will attach standard errors from the θ covariance, since each readout is linear in θ. This gives concrete targets to compare against the model's current −0.39, +0.15, −0.66 and −0.24.

```python
def readout_truth(de_df, edit, D, th, mult, Vth=None):
    rr = de_df[(de_df.edit==edit)&(de_df.reached)]
    Wm = np.zeros((len(rr), len(th)))
    for k,(sid_, t) in enumerate(zip(rr.stay_id.values, rr.hour.values.astype(int))):
        i = pos[sid_]
        for s_ in range(12, t):
            L = t - s_
            if L <= len(th): Wm[k, L-1] += (mult-1)*D[i, s_]
    w = Wm.mean(0)
    val = w @ th
    se = np.sqrt(w @ Vth @ w) if Vth is not None else np.nan
    return val, se, len(rr)
th_ins = b_ins[:12]; V_ins = Vc_ins[:12,:12]; th_ne = bn[:12]; V_ne = Vn[:12,:12]
for ed, mult, D, th, Vv in [('insulin_x2',2.0,INS_tu,th_ins,V_ins),('insulin_half',0.5,INS_tu,th_ins,V_ins),
                             ('ne_x2',2.0,NE_tu,th_ne,V_ne),('ne_half',0.5,NE_tu,th_ne,V_ne)]:
    v, se, n = readout_truth(de, ed, D, th, mult, Vv)
    print(f"{ed:13s} truth-estimate {v:7.2f} (se {se:.2f})  model original {de[(de.edit==ed)&de.reached]['shift'].mean():.3f}  rows {n}")
```