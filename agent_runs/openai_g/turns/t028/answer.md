The recovered results support the current repair, but the timing and rate-support checks have not run yet. I’ll request just those diagnostics. Correct alignment and adequate dose variation are necessary before interpreting the remaining norepinephrine estimates as causal.

```python
print("CHANNELS:", events)
print("ARRAY HOURS:", hour[0].tolist())
print("FIRST STAY:")
print(pd.DataFrame(arr[0], columns=events).round(3).to_string())
print("TREATMENT HOURS:", np.unique(treatment_hours_v2).tolist())
print("FIRST 24 NE EXPOSURES:", A_ne_v2[:24].tolist())
print("NUISANCE SETTINGS:", nuisance_settings_v2)
for _label, _a in [("insulin", A_insulin_v2), ("norepinephrine", A_ne_v2)]:
    _q = _a[np.isfinite(_a) & (_a > 0)]
    print(_label, "positive n", len(_q),
          "quantiles", np.quantile(_q, [0, .01, .1, .5, .9, .99, 1]).tolist(),
          "frequent rates", pd.Series(_q).value_counts().head(5).to_dict())
print("TUNING IDS:", tuning_ids.tolist())
with open("/work/notes.md", "a") as _f:
    _f.write("\n\n## Restart recovery, runs 20–22\n"
             "Recovered the existing three-tensor null-dose edit. Earlier rollouts were exactly "
             "invariant for all four dose edits, and paired factual NLL on 200 tuning stays "
             "changed by -0.0000078 nats. Continuing causal checks of late NE signals before "
             "accepting the null effect; remaining 200 tuning stays still need factual validation.\n")
```