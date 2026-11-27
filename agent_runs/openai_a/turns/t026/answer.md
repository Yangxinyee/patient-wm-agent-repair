The doses are rounded, and norepinephrine is capped at 1. That may provide useful treatment-policy discontinuities, but rounding alone is not an instrument. I’ll check how rates depend on recorded readings and inspect the value-head inputs. Clear dose jumps without corresponding pre-treatment outcome jumps would support a discontinuity-based estimate.

```python
from scipy import stats
from pathlib import Path

print("=== NORMALIZATION FORMAT ===")
for entry in VOCAB["numeric"]:
    print({k: v for k, v in entry.items()
           if "edge" not in k and "centre" not in k and "bin" not in k})
print("First numerical entry keys:", list(VOCAB["numeric"][0]))
print("Config:", VOCAB["config"])
print("=== NON-CORE HEAD / EMBEDDING WEIGHTS ===")
for name in NAMES:
    if name.startswith(("embed.", "encode_action.", "attr_heads.value_bin.",
                        "last_", "query_norm.", "z_prior.", "z_head.",
                        "gap.", "type_emb.")):
        print(name, SHAPES[name])

P = pa.reshape(len(train_ids), len(hour_grid), 6)
columns = list(panel.columns)
print("\n=== DOSE / READING RELATIONSHIPS ===")
for drug, reading in [("insulin", "glucose_finger"),
                      ("norepinephrine", "map_arterial"),
                      ("norepinephrine", "map_cuff")]:
    d = panel[drug].to_numpy()
    r = panel[reading].to_numpy()
    mask = np.isfinite(r)
    grouped = pd.DataFrame({"reading": r[mask], "dose": d[mask]}).groupby("reading").dose.agg(["mean", "count"])
    print(drug, "by", reading)
    print(grouped.loc[(grouped.index >= (100 if drug == "insulin" else 45)) &
                      (grouped.index <= (205 if drug == "insulin" else 85))].round(4).to_string())
    pos = d[d > 0]
    print("Most frequent positive doses:",
          pd.Series(pos).value_counts().head(12).to_dict())
print("\n=== BASELINE SUMMARY FILES ===")
for k in (1, 2, 3, 4):
    print(k, Path(f"/work/fwd_{k}/summary.json").read_text())
```
