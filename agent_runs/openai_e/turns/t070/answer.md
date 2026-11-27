The retained objects include separate synthetic contrasts for doubling and halving, plus an admission-null constraint. I’ll check their units and the causal coefficients next. If the halving contrast is disproportionately suppressed, I can rebalance the feature instead of simply scaling an already asymmetric edit.

```python
import inspect
import numpy as np

print("CAUSAL FITS")
print("insulin dynamic parameters:", np.asarray(dyn_fit["x"]))
print("NE_DML:", repr(NE_DML)[:1200])
print("panel drug distributions:")
print(panel[["insulin", "norepinephrine"]].describe().to_string())
print("panel columns:", panel.columns.tolist())

print("\nTARGETED HELPER SOURCE")
for function62 in [feature_metrics57, full_insulin_slope58, nuisance_features]:
    print("\n", function62.__name__)
    try:
        print(inspect.getsource(function62)[:2600])
    except Exception as error62:
        print("Source unavailable:", type(error62).__name__)
        print("Referenced names:", function62.__code__.co_names)
        print("Constants:", repr(function62.__code__.co_consts)[:900])

print("\nCONTRAST UNITS")
for key62 in prelude_contrasts46:
    a62 = np.asarray(prelude_contrasts46[key62])
    b62 = np.asarray(slot_deltas47[key62])
    y62 = np.asarray(targets57[key62])
    d62 = np.asarray(dose_contrasts46[key62])
    print(key62,
          "prelude", a62.shape, "RMS", float(np.sqrt(np.mean(a62*a62))),
          "slot_delta", b62.shape, "RMS", float(np.sqrt(np.mean(b62*b62))),
          "target", y62.shape, "mean", np.mean(y62, axis=0),
          "dose", d62.shape, "mean", np.mean(d62, axis=0))
print("raw E RMS:", float(np.sqrt(np.mean(raw_E57**2))))
print("slot feature RMS:", float(np.sqrt(np.mean(slot_features47**2))))
print("scale57 range:", np.quantile(scale57, [0, .5, 1]))
print("raw candidate metrics:",
      feature_metrics57(constrained_fits57[("zero", 10.0)]))
print("update-direction norms:",
      {k: float(np.linalg.norm(v)) for k, v in bin_directions46.items()})

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 62: unit and identification audit\n"
            "Inspecting exact causal parameters and contrast units before "
            "balancing insulin dose directions or constructing a NE edit. "
            "No weight changes. Python runs: 62; forwards: 7.\n")
```
