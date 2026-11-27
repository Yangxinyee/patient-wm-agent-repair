Run 84 narrows down the bias:
- **Ruled out:** `last_status` (row 0 is zero padding), the class-0 handling, and the 4th `last_numeric` input (it must be 1).
- **Confirmed:** `type_emb` row 1 is the right one.
- **What remains:** the register age. Age in minutes gives a mean error of −1.8 and age in hours gives +0.7. Interpolating, the error vanishes at a log-age input of about 0.17. That is close to log(1+5)/10 = 0.179: the age in minutes measured at the state's time, where the action bundle comes 5 minutes after the reading, rather than at the predicted reading's time.

Next I'll test that version directly, alongside a scan of constant log-age inputs.
- **Confirms it:** the age-at-state-time version brings the mean error to about 0 and the median |error| near the 1.0 sampling floor.
- **Refutes it:** the error stays well above that floor.

```python
pre2 = []
for j, x in zip(idx, pre):
    d = DH.iloc[j]; i0 = int(d.i0)
    ph = PREV.get((d.stay_id, round(d.hour,4)))
    ages = (SHR[i0]-ph)*3600.0 if (ph is not None and ph==ph) else None
    pre2.append(x[:4] + (ages,) + x[5:])
def score2(P, **kw):
    e = np.array([pred(*x[:5], **kw) - x[5] for x in P]); return float(np.median(np.abs(e))), float(e.mean()), e
a, b, e_st = score2(pre2)
print('age at state time (min):', round(a,3), round(b,3))
print('state-age values (min) sample:', np.unique(np.round([x[4]/60 for x in pre2 if x[4] is not None]),return_counts=True))
# constant scan of num[1]
def pred_c(ctx, mu, ls, lv, age, c1):
    return pred(ctx, mu, ls, lv, (math.exp(10*c1)-1)*60.0 if lv is not None else None)
for c1 in [0.0, 0.1, 0.15, 0.18, 0.22, 0.3]:
    e = np.array([pred_c(*x[:5], c1) - x[5] for x in pre]); print('const', c1, round(float(np.median(np.abs(e))),3), round(float(e.mean()),3))
```