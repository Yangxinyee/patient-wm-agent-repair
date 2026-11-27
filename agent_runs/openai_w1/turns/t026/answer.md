The kernel retains the hourly panels, adjusted dose innovations, and representative-state sample, so those analyses need not be repeated. I’ll now retrieve the baseline rollout shifts and exact normalisation tables. This will tell us whether the model’s problem is sign, magnitude, or timing, and provide the units needed for a weight edit.

```python
print("=== ORIGINAL ROLLOUT ===")
print(baseline_rollout.head(6).to_string(index=False))
print("columns:", baseline_rollout.columns.tolist())
groupcols = [c for c in baseline_rollout.columns
             if baseline_rollout[c].dtype == object]
print("groups:", groupcols)
if groupcols:
    print(baseline_rollout.groupby(groupcols).mean(numeric_only=True).to_string())
print("evaluation stay IDs:", eval_ids)

print("=== NUMERIC VOCAB ===")
print("codes:", VOCAB["codes"])
print("config:", VOCAB["config"])
print("numeric:", json.dumps(VOCAB["numeric"])[:3700])
print("numeric_note:", VOCAB.get("numeric_note"))
print("=== REMAINING RELEVANT SHAPES ===")
for name in NAMES:
    if name.startswith(("z_", "filter_", "pending_", "embed_transition", "embed.kq", "encode.")):
        print(name, SHAPES[name])

print("=== PANEL / ROBUSTNESS LAYOUT ===")
print("hours:", hours.tolist())
for key in ["insulin", "norepinephrine"]:
    print("rates", key, R[key].shape, R[key][0].tolist())
for key in ["glucose_finger", "map_arterial"]:
    print("panel", key, panel[key].shape, panel[key][0].tolist())
for key, result in repair_robust.items():
    print(key, {k: (v.shape if isinstance(v, np.ndarray) else v)
                for k, v in result.items()})
print("feature names:", repair_feature_names)
print("=== EARLY NOTES ===")
print(open("/work/notes.md").read()[:2200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 21\nRecovered existing hourly panels, cross-fitted effect diagnostics, "
            "representative state sample, and original rollout table. Retrieving quantitative "
            "baseline and normalization before constructing a reversible edit. No edits.\n"
            "Counts: 21 Python runs, 4 forward passes.\n")
```
