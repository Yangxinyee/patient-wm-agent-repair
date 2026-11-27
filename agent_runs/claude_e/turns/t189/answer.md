The rate-level readout on held-out arterial rows with NE running is moderate: corr 0.70 (R2 0.48), but gain only 0.38. Made unit-gain, it would be scaled up about 2.6 times, noise included.

lda_in returns a tuple, not a dict, so I need its layout before building the NE-running gate. I'll print what the tuple holds and then rerun the held-out gate check.

```python
Gh_ = lda_in(ne_run_tr, tm7 & ~hn_tr)
print(type(Gh_), len(Gh_), [type(x).__name__ for x in Gh_], [np.shape(x) for x in Gh_])
Gh = Gh_[0] if isinstance(Gh_[0], dict) else dict(zip(['w','c','sep'], Gh_))
print({k: (np.shape(v) if np.ndim(v) else float(v)) for k,v in Gh.items()})
sc = WCs.astype(np.float64) @ np.asarray(Gh['w'], float) - float(Gh['c'])
print('held-out hn-gate score: ma&NE pct1/5 %s | ma&noNE pct95/99 %s' % (np.percentile(sc[:, ma_run_ho], [1,5]).round(2), np.percentile(sc[:, (cs==7)&~pv_ne], [95,99]).round(2)))
print('mismatch pv_ne vs NE recorded at h-1 on ma rows: %.3f' % ((pv_ne != (rne1_ho>0))[cs==7]).mean())
```