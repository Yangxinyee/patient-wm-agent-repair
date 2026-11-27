My replica's means differ from the tool's by sd 4.4 mg/dL. Worse, the difference correlates with the lag-1 insulin innovation, at +0.66 ± 0.19 mg/dL per u/h. So replica-only residual regressions would be biased at lag 1, though not at lags 2–3.

The efficient fix is a two-step estimate:
1. Estimate the missing dose effect from replica residuals on thousands of training stays (precise).
2. Correct it with the tool-minus-replica coefficient, which has far lower noise per stay than raw residuals because that difference has sd 4.4 rather than 12. setA already gives one such estimate.

There is also a design point for NE. Its doses persist through titration, so in the x2 rollout (titration switched off) an innovation-based correction would make the effect fade too fast. For MAP I'll therefore regress residuals on NE rate levels, alongside innovations, to compare.

This run generates replica samples for 3,500 more training stays (6,000 in total), computes replica residuals, and regresses:
- finger-glucose residuals on insulin innovations at lags 1–10;
- arterial-MAP residuals on NE rate levels at lags 1–8.

```python
P3 = Ptr3; Rm = P3.R.fillna(0).values.reshape(-1, 26); NUNm = P3.nuN.fillna(0).values.reshape(-1, 26)
more = [s for s in rng.choice(S_ord, 5000, replace=False) if s not in set(st.tolist())][:3500]
grp_more = {s: g_ for s, g_ in tr[tr.stay_id.isin(more)].groupby('stay_id', sort=False)}
t0 = time.time(); fsM = gen_many(grp_more, more); print('gen %.0fs' % (time.time()-t0))
allrows = fsF + fsM
keep = [r for r in allrows if r[2] >= 0 and r[0] in ('glucose_finger', 'map_arterial')]
mu_all, _ = head_mean(keep)
code_ = np.array([r[0] for r in keep]); y_ = np.array([r[4] for r in keep]); res_all = y_ - mu_all
rs_all = np.array([six[r[1]] for r in keep]); rt_all = np.array([int(r[2]) for r in keep])
pickle.dump({'meta': [(r[0], r[1], r[2], r[3], r[4]) for r in keep], 'X': np.stack([r[8] for r in keep]).astype(np.float32), 'mu': mu_all}, open('/work/rep_samples.pkl', 'wb'))
def regr(y, Xl, names):
    Xr = np.column_stack([np.ones(len(y)), Xl]); b = np.linalg.lstsq(Xr, y, rcond=None)[0]
    s2 = (y - Xr@b).var(); se = np.sqrt(np.diag(np.linalg.inv(Xr.T@Xr))*s2); return b[1:], se[1:]
fm = code_ == 'glucose_finger'; am = code_ == 'map_arterial'
b, se = regr(res_all[fm], lagm(NUm, rs_all[fm], rt_all[fm], 10), None)
print('FINGER n %d resid sd %.2f | coef on ins nu lags1-10:' % (fm.sum(), res_all[fm].std()), b.round(2), '| se ~%.2f' % se.mean())
b2, se2 = regr(res_all[am], lagm(Rm, rs_all[am], rt_all[am], 8), None)
print('ART n %d resid sd %.2f | coef on NE rate lags1-8:' % (am.sum(), res_all[am].std()), b2.round(2), '| se ~%.2f' % se2.mean())
b3, se3 = regr(res_all[am], lagm(NUNm, rs_all[am], rt_all[am], 8), None)
print('ART coef on NE innovation lags1-8:', b3.round(2), '| se ~%.2f' % se3.mean())
```