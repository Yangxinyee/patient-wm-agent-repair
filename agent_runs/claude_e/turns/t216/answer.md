On tuning batch 2020000100–199, `ins_only(−2.6)` costs **+0.0043 per reading** (1.62044 → 1.62471), leaving only 0.0007 of the 0.005 budget:

| Rows | NLL change per row | Total (nats) |
|---|---|---|
| glucose_finger | +0.0126 | 17.2 |
| lab glucose | +0.0061 | 3.1 |
| map_arterial and map_cuff | about 0 | about 0 |

The finger figure matches what the cost curve predicted. The total is higher than the sel1 offline 0.0037, so −2.6 is too risky for an unseen set. A gain of about −2.4 should cost about 0.0036 but would leave insulin ×2 near −12 against truth −13.2.

The readout itself is already the best linear one, so the only ways out are lower cost or a bigger response per unit of gain. One candidate does both. Earlier, the model's dependence on the last finger reading looked weaker than the true dependence (about 0.84–0.97 against 1.05). If its persistence is genuinely too weak, a correction would improve factual NLL and also let a dose shift carry forward further. That would allow a smaller insulin gain for the same response.

I'll test this on the sel1 finger rows. The test regresses the original model's residuals on the previous reading within tiers, plus the true dose deviation. A clearly positive within-tier slope would confirm weak persistence; a slope near zero would rule out the idea.

```python
s6 = R1[R1.code==6].copy()
hh6 = np.floor(s6.hour.values.astype(float)).astype(int) - 1
s6['gp'] = [gget2(gfI, s, h) for s, h in zip(s6.stay.values, hh6)]
s6 = s6[~np.isnan(s6.gp)]
tt = tier5(s6.gp.values)
cols6 = []
for k in range(5):
    mk = (tt==k).astype(float); cols6 += [mk]
gpc = (s6.gp.values - 180)/50
X6 = np.column_stack(cols6 + [gpc, gpc**2, s6.dev1t.values])
b6, se6 = ols_se(X6, s6.res.values.astype(float))
print('gf rows n', len(s6), '| resid ~ tierFE + gp: slope gp(per 50) %.3f (se %.3f), gp^2 %.3f (se %.3f), dev1t %.3f (se %.3f)' % (b6[5], se6[5], b6[6], se6[6], b6[7], se6[7]))
for k in range(4):
    mk = tt==k
    if mk.sum() > 50:
        bb, ss = ols_se(np.c_[np.ones(mk.sum()), s6.gp.values[mk]], s6.res.values[mk].astype(float))
        print(' tier', k, 'n', int(mk.sum()), 'resid slope on gp %.3f (se %.3f)' % (bb[1], ss[1]))
```