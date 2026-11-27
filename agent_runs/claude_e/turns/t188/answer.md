On the 38 shared stays my offline evaluator matches the tool closely: glucose_finger +0.0086 vs +0.0087 per row, map_arterial +0.0091 vs +0.0089, overall +0.0053 vs +0.0051. I can now score edits offline on the 200 held-out stays (100 tuning in sel1, 100 training in A).

The per-code differences from the tool's random stays come from stay mix, which I can reweight later. Most of the cost is the old NE pair, applied to every arterial row, and it doesn't produce the x2 response anyway.

Next I'll build the replacement NE readout for the rate level, fit only on arterial rows where NE ran in the previous bundle, since only those will be gated on. I'll check its held-out R2 and gain on the same rows in WC. Then I'll fit an "NE in previous bundle" gate and check it separates held-out arterial rows cleanly, as the insulin gate does.

```python
hn_tr = A_.hn.values == 1
ne_run_tr = tm7 & hn_tr
w_l, b_l = fit_lin(ne_run_tr, y_ne, ~tm7, alpha=1e-3, ndr=4)
rne1_ho = np.array([ (GridFor(s)['ne'].at[s,int(h)-1] if (s in GridFor(s)['ne'].index and int(h)-1 in GridFor(s)['ne'].columns and pd.notna(GridFor(s)['ne'].at[s,int(h)-1])) else 0.0) for s,h in zip(RR.stay.values, RR.hour.values)])
ma_run_ho = (cs==7) & pv_ne
rh = WCs[:, ma_run_ho].astype(np.float64).mean(0) @ w_l + b_l
yh = rne1_ho[ma_run_ho]
c_ = np.corrcoef(rh, yh)[0,1]
print('held-out NE-running arterial rows %d: corr %.3f R2 %.3f gain %.3f | rate mean %.3f sd %.3f' % (ma_run_ho.sum(), c_, c_**2, np.cov(rh,yh)[0,1]/yh.var(), yh.mean(), yh.std()))
Gh = lda_in(ne_run_tr, tm7 & ~hn_tr)
print('lda gate keys', list(Gh.keys()), 'sep', Gh.get('sep'))
sc = WCs.astype(np.float64) @ np.asarray(Gh['w'], float) - Gh['c']
print('held-out hn-gate score: ma&NE pct1/5 %s | ma&noNE pct95/99 %s' % (np.percentile(sc[:, ma_run_ho], [1,5]).round(2), np.percentile(sc[:, (cs==7)&~pv_ne], [95,99]).round(2)))
print('mismatch pv_ne vs NE recorded at h-1 on ma rows: %.3f' % ((pv_ne != (rne1_ho>0))[cs==7]).mean())
```