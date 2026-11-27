The model's one-step coefficients on insulin dose deviation are far too weak, which supports the head-correction plan. These come from the same regression specification with history controls:

| lag (h) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| true (train, 30k stays, SE ≈ 0.045) | −2.04 | −1.23 | −0.25 | +0.38 | +0.80 | +0.69 |
| model (200 tuning stays, SE ≈ 0.12) | −0.34 | +0.13 | +0.26 | +0.11 | +0.12 | −0.10 |
| Δβ = true − model | −1.71 | −1.36 | −0.51 | +0.27 | +0.68 | +0.79 |

The tuning-sample truth is noisy (SE ≈ 0.55) but consistent at lags 1–2.

Δβ is my first estimate of the per-lag correction. One thing still worries me: the within-tier slopes on the current reading also differ (model 0.97 / 0.96 / 0.84 vs truth about 1.05). If the model carries a glucose shift forward less persistently than the true process, correct one-step terms could still under-accumulate in the rollout. I can't interpret the "extra" coefficients because I don't know their layout.

Next I'll record this in the notebook. Then I'll run my own regression with explicit, named columns on the same rows, for truth and model. That lets me compare dose terms and glucose-persistence terms properly, and check whether Δβ depends on the specification. If Δβ stays similar with and without the extra lags, the "missing linear dose term" hypothesis gains support.

```python
with open('/work/notes.md','a') as fh:
    fh.write("""
## Runs 85-90 (after restart 3)
- Previous reply was cut by a safety filter; kept strictly to allowed methods (records/state regressions, closed-form edits, tool checks).
- Kernel still holds: X (sel1 states 5116x992), RR (10353 reading rows sel1 with local pm), WC/HID/LG (8 z-draws), Dtr2 (train one-step design),
  build_ins_design2(G, stays) -> design (s = position in stays list!), ins_regress2(D, ycol) -> (coef, se), layout 15 tier coefs + 10 extras + dev1..6.
- One-step dev coefs (same spec, extras): TRUE train [-2.04 -1.23 -0.25 0.38 0.80 0.69] (se .045); MODEL pm on 200 tuning stays (f3 + sel1)
  [-0.34 0.13 0.26 0.11 0.12 -0.10] (se .12). Delta = [-1.71 -1.36 -0.51 0.27 0.68 0.79]. Model tier slopes on g0 weaker (0.97/0.96/0.84 vs ~1.05).
""")
def my_reg(D, ycol):
    D = D[D[ycol].notna() & D.g0.notna()].copy()
    t = D.tier.values.astype(int); g0 = D.g0.values.astype(float); n = len(D)
    cols, names = [], []
    for k in range(5):
        m = (t == k).astype(float)
        if m.sum() < 30: continue
        c = np.median(g0[t == k])
        cols += [m, m*(g0-c)/50, m*((g0-c)/50)**2]; names += [f't{k}', f't{k}_g0', f't{k}_g0sq']
    for v in ['g1','g2','g3','lab','lab1']:
        x = D[v].values.astype(float); miss = np.isnan(x)
        cols += [np.where(miss, 0, x - g0)/50, miss.astype(float)]; names += [f'{v}-g0', f'{v}_miss']
    for k in range(1,7):
        cols.append(D[f'dev{k}'].values.astype(float)); names.append(f'dev{k}')
    Xm = np.column_stack(cols); y = D[ycol].values.astype(float)
    b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    r = y - Xm @ b; s2 = r.var()*n/(n-Xm.shape[1])
    se = np.sqrt(np.diag(np.linalg.pinv(Xm.T @ Xm))*s2)
    return pd.Series(b, names), pd.Series(se, names), np.sqrt(s2)
bt_, st_, rt_ = my_reg(Dtr2, 'y'); by_, sy_, ry_ = my_reg(Dm, 'y'); bm_, sm_, rm_ = my_reg(Dm, 'pm')
tab = pd.DataFrame({'true_tr': bt_, 'se_tr': st_, 'true_tu': by_, 'se_tu': sy_, 'model_tu': bm_, 'se_m': sm_})
pd.set_option('display.width', 200)
print(tab.round(3).to_string())
print('resid sd: train', rt_, 'tuning y', ry_, 'tuning pm', rm_)
print('Delta dev (true_tr - model):', (bt_[[f'dev{k}' for k in range(1,7)]] - bm_[[f'dev{k}' for k in range(1,7)]]).round(2).values)
```