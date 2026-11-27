The new cells make the no-fresh-hour innovation unpredictable (R² 0.0009, from 0.034). The norepinephrine response at short lags is stable:

| lag (h) | effect (mmHg per mcg/kg/min) |
|---|---|
| 1 | 2.9–3.4 |
| 2 | 2.7–2.9 |
| 3 | 1.6–2.0 |
| 4 | 0.8–1.1 |
| 5 | 0.4–0.6 |
| 6 and later | about 0 |

The negative first-in-window coefficient is smaller but still there (−0.6 to −1.1). The sum of the coefficients is nearly the same in every window, though: about 8.8–8.9 for windows of 6, 8 and 12 hours. So the steady-state gain is well identified even though the long-lag split is not. Norepinephrine rates are very persistent, so a single innovation moves all later rates in the window, and the first coefficient becomes a small difference of large numbers that soaks up any misspecification. What the readout needs is cumulative partial sums of the response, and those are stable. I still need to check for nonlinearity later, especially because ×2 pushes norepinephrine above the observed maximum of 1.0.

Now I need to understand the model itself, because the form an edit can take limits what effect structure I can implement. I will record the findings in the notebook, then list all 181 tensors with their shapes, and read the vocab config, codes, normalisation and bin tables for the drugs and readings.

```python
notes_add = """
## Runs 15-26: model readout + norepi
- dose_edit readout rows = hours 13-24 with a RECORDED glucose_finger (insulin) / map_arterial (norepi) reading; 'reached' = after
  first dose at/after hour 12. Mean over reached rows. Norepi rows only exist in arterial-line stays (~half of stays, stay-level).
- Original model on sel100 tuning stays (fwd_1): insulin x2 -0.39, x1/2 +0.15; ne x2 -0.66, ne x1/2 -0.24  (tiny / wrong sign).
- Prelim insulin truth on same rows (theta: lag1 -2.2, lag2-4 -2.4, lag5 -1.5, 6 -1.1, 7 -0.9, 8 -0.8, 9 -0.6, 10 -0.5, 11 -0.3, 12 -0.1):
  x2 = -17.5, x1/2 = +8.8 (linear assumption).
- Norepi POLICY: start only when off & fresh MAP<=~64 at ~0.05; when on, multiplicative titration r_t = r_{t-1}*f(MAP)*noise, clip [0.01,1];
  stop w.p. ~0.15-0.2 if MAP>75. Uses arterial if present else cuff; in no-fresh hours uses LAST MAP. Innovation (r - E[r|cell]) unpredictable
  from history once cells include (last MAP bin, age) for no-fresh hours. Cuff = art + 3 (sd 6.7). Arterial present 95% at random in line stays.
- Norepi IV impulse response on map_arterial per 1 mcg/kg/min: lag1 ~3.0-3.4, lag2 ~2.7-2.9, lag3 ~1.6-2.0, lag4 ~0.8-1.1, lag5 ~0.5, later ~0;
  first-in-window coefficient negative (-0.6..-1.1) = artifact of persistent rates; SUM of lags stable ~8.8 across windows.
"""
open('/work/notes.md','a').write(notes_add)
V = json.load(open('/model/vocab.json'))
print(V.keys())
print(json.dumps(V['config'], indent=0)[:3000])
print("codes:", V['codes'])
for k in V:
    if k not in ('config','codes'):
        s = json.dumps(V[k]); print(k, ":", s[:1500])
```