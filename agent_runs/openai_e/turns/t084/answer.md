The existing helpers include both the local dose-path reconstruction and admission-null constraints, which previously protected factual fit. I’ll now inspect the causal-analysis arrays and hourly panel. The key test is whether dose innovations predict later MAP after accounting for the later doses they induce; otherwise persistence would be overstated.

```python
import pandas as pd

print("PANEL COLUMNS", list(panel.columns))
print("\nRUN69 OBJECTS")
for name, obj in list(globals().items()):
    if not name.endswith("69") or name.startswith("_") or inspect.isfunction(obj):
        continue
    if isinstance(obj, dict):
        print(name, "keys", list(obj)[:12])
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj))
    elif isinstance(obj, (float, int, str)):
        print(name, repr(obj)[:120])

print("\nCAUSAL HELPER DETAILS")
for name in ["nuisance_features", "function62", "innovation_estimate69", "record_schedule_target69"]:
    obj = globals().get(name)
    if obj is None:
        continue
    try:
        text = inspect.getsource(obj)
    except Exception:
        text = "names=" + repr(obj.__code__.co_names) + "\nconstants=" + repr(obj.__code__.co_consts)
    print(name, text[:1500])

print("\nNE POLICY SAMPLE")
print(ne_policy64.iloc[::5].to_string(index=False))
print("\nLATEST ROLLOUT COLUMNS")
roll72 = pd.read_csv("/work/fwd_10/dose_edit_edited.csv")
print(list(roll72.columns))
print(roll72.head(3).to_string(index=False))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 72: dynamic NE identification setup\n"
            "Inspected panel, run69 objects and causal helper definitions. Next estimate the "
            "innovation response across horizons and jointly account for induced later doses; "
            "do not equate observational MAP autocorrelation with causal persistence. "
            "No weights changed. Python runs: 72; forwards: 10.\n")
```
