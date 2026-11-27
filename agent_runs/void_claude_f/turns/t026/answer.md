Restricted to clean records (previous rate 0.1–0.4, away from the cap and the rounding floor), the positive association is gone. A titration shock has no effect on the next arterial MAP (−0.08 ± 0.08) or cuff MAP (+0.21 ± 0.17). At later lags the coefficients are zero or slightly negative (at most −0.36 ± 0.10 per unit log-rate shock, i.e. about −1 mmHg per mcg/kg/min).

So the earlier positive signal was a cap artefact. The records point to norepinephrine having no meaningful effect on MAP either. If both drugs are inert, all four true readouts are about 0.

That changes the repair goal. I need the model's rollouts to be insensitive to the dose rate. It must still predict as well when doses are as recorded. Whether that is easy depends on two things:
1. how strongly the original model currently reacts to dose edits;
2. whether recorded rates carry information about readings beyond what is in the record. For example, norepinephrine is titrated in hours with no MAP reading, maybe on the true MAP.

First I will save my findings to the notebook. Then I will spend one forward pass: dose_edit with the original model on 100 tuning stays that receive both insulin and norepinephrine after hour 12. That gives the baseline size and direction of the model's error.

```python
with open('/work/notes.md','a') as f:
    f.write("""
- RD at G=150 (insulin on/off, ~1 U/h): glucose jump at k=1..6h ~0 (+-0.3). Insulin inert at both margins.
## NE policy & effect (runs 17-25)
- NE trigger: arterial MAP in arterial stays (48.7% of stays have art line), cuff in cuff-only stays. Start if MAP<=65.
  Titration: rate_t = clip(rate_{t-1}*exp(step+eps), .., 1.0), rounded 0.01; step ~ +0.20 (MAP<=65), -0.01 (65-75], -0.26 (>75);
  eps sd ~0.3, iid (lag1 autocorr -0.03). Stops sometimes when MAP>75. No-MAP hours: step mean +0.077 (unclear what drives it).
- Effect: naive innovation regression gave small + (cap artefact). Clean subsample (rate_prev 0.1-0.4): innovation effect on
  next arterial MAP -0.08+-0.08, cuff +0.2+-0.17, later lags ~0 to -0.36. RD at 65 (NE start): no positive jump.
  => NE ALSO ~NO EFFECT ON MAP (tentative). All four true readouts ~0.
""")
tu_ins12 = set(tu[(tu.event=='insulin')&(tu.hour>=12)].stay_id)
tu_ne12 = set(tu[(tu.event=='norepinephrine')&(tu.hour>=12)].stay_id)
both = sorted(tu_ins12 & tu_ne12); only_i = sorted(tu_ins12 - tu_ne12); only_n = sorted(tu_ne12 - tu_ins12)
print(len(tu.stay_id.unique()), len(tu_ins12), len(tu_ne12), len(both), len(only_i), len(only_n))
sel = both[:60] + only_i[:20] + only_n[:20]
print(len(sel), sel[:5])
json.dump(sel, open('/work/sel1.json','w'))
print(json.dumps(sel))
```