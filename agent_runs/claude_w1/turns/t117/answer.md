That result cannot be right as it stands. The ridge fits explain essentially none of the dose variance (residual sd equals raw sd at every lag). Yet doses track glucose, and the head input encodes the glucose history almost perfectly. The likeliest cause is misalignment between my lagged-dose matrix (built from the panel) and the sample rows.

`gen_samples` returns its own lag-1..3 doses in each row, so I'll check them against my `lagm` values directly. I'll also re-run the lag-1 regression using `gen_samples`' own dose column. If the two dose versions disagree, the bug is in my index or hour mapping. If they agree but R² is still ~0, the vectors themselves are misaligned with their rows.

```python
gl1 = np.array([r[5] for r in fsF])[m]; gl2 = np.array([r[6] for r in fsF])[m]
print('corr(gen lag1, my LD lag1) %.3f, (gen lag1, my lag2) %.3f, (gen lag2, my lag2) %.3f' % (np.corrcoef(gl1, LD[:,0])[0,1], np.corrcoef(gl1, LD[:,1])[0,1], np.corrcoef(gl2, LD[:,1])[0,1]))
print('mean abs diff gen-my lag1: %.3f' % np.abs(gl1 - LD[:,0]).mean(), '| sd gen lag1 %.3f' % gl1.std())
res1 = cvres(xF, gl1); print('ridge on gen lag1: resid sd %.3f of sd %.3f' % (res1.std(), gl1.std()))
curv = np.array([r[4] for r in fsF])[m]; resc = cvres(xF, curv); print('ridge on cur value: resid sd %.2f of sd %.2f' % (resc.std(), curv.std()))
print('example rows (stay, hour, gen lag1, my lag1):', [(int(S_ord[rs[i]]), int(rt[i]), gl1[i], LD[i,0]) for i in range(20, 26)])
```