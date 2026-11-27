The earlier regressions suggest lower glucose after higher insulin doses and higher MAP after higher norepinephrine doses. Those associations do not yet establish the drugs’ effects: dose selection and measurement noise could bias them. I’ll inspect timing and dose histories before testing pre-dose placebos and delayed responses.

```python
print("Panel hours:", hours)
for label, panel in [("P", P), ("R", R), ("L", L), ("Age", Age)]:
    print(label, [(k, np.shape(v)) for k, v in panel.items()])
print("=== TWO EXAMPLE STAYS ===")
for sid in stay_ids[:2]:
    ss = train.loc[train.stay_id == sid, ["hour", "event", "value", "rate", "dose"]]
    print("stay", sid)
    print(ss.loc[(ss.hour >= 9) & (ss.hour <= 15)].to_string(index=False))
print("=== RESIDUAL CONTAINER ===")
for key, val in causal_residuals.items():
    print(key, type(val).__name__)
    if isinstance(val, dict):
        print([(k, type(v).__name__, np.shape(v)) for k, v in val.items()])
    elif isinstance(val, (tuple, list)):
        print([(type(v).__name__, np.shape(v)) for v in val])
print("=== DOSE AND MEASUREMENT SUMMARY ===")
for e in ["insulin", "norepinephrine", "glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    ss = train.loc[train.event == e]
    vv = ss.rate if e in R else ss.value
    print(e, "rows", len(ss), "stays", ss.stay_id.nunique(),
          "hours", ss.hour.min(), ss.hour.max(),
          "quantiles", np.round(vv.quantile([0,.1,.5,.9,1]).values, 4))
print("=== FORWARD FILE CONTENTS ===")
for folder in ["/work/fwd_1", "/work/fwd_2", "/work/fwd_3"]:
    print(folder, os.listdir(folder))
print("=== PRIOR ESTIMATOR SETTINGS ===")
print("params", params if "params" in globals() else None)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 15\nInspected panel indexing, examples, and saved dose-innovation residuals. Next priority is causal dynamics and properly residualized pre-dose placebos, not more uncontrolled head-reconstruction grids.\nCounts: 15 Python runs, 3 forward passes.\n")
```