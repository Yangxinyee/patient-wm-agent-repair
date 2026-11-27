Readings run from hour -1 to 24 and infusion records from hour 0 to 24. Glucose_finger is present at about half the hours, rising over the stay. Glucose_lab is at 20% and map_arterial at a steady 46%. Insulin use rises from 31% to 42% of stays across the stay; norepinephrine stays near 32%.

Before estimating effects I need the dosing policy: what drives the clinicians' choices. That decides which confounders a regression must adjust for. Next I'll check:
- whether measurement patterns are per-stay (e.g. arterial-line patients),
- how insulin starts, stops and changes relative to the latest glucose and the previous rate,
- the same for norepinephrine against MAP.

If doses are a clean, noisy function of the latest readings, adjusting for those readings should identify the effects. If dosing looks driven by something not recorded, I'll need another identification strategy.

```python
G = {k: P[k].values for k in P}   # arrays [n_stays, 28], col j <-> hour H[j]
hrs = list(H)
def col(hr): return hrs.index(hr)
# per-stay measurement patterns
for k in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    f = np.isfinite(G[k][:, col(-1):col(24)+1]).mean(1)
    print(k, "per-stay presence frac quantiles:", np.round(np.quantile(f,[0,.1,.25,.5,.75,.9,1]),2))
ins = np.nan_to_num(G['insulin']); ne = np.nan_to_num(G['norepinephrine'])
# transitions of insulin on/off
on = ins[:, col(0):col(24)+1] > 0
print("insulin: frac stays ever on", on.any(1).mean(), " once on, stays on? frac of off after on:",
      np.mean([ (~r[np.argmax(r):]).mean() for r in on if r.any()]))
onn = ne[:, col(0):col(24)+1] > 0
print("norepi: frac stays ever on", onn.any(1).mean(), " off after on frac:",
      np.mean([ (~r[np.argmax(r):]).mean() for r in onn if r.any()]))
print("insulin start hour dist:", np.bincount(np.argmax(on[on.any(1)],1)))
print("norepi start hour dist:", np.bincount(np.argmax(onn[onn.any(1)],1)))
```