The records give fairly consistent evidence that insulin lowers glucose, including a threshold-based estimate. Norepinephrine is less settled: its estimated effect changes substantially with adjustment, so I won’t treat one regression as definitive. I’ll inspect the cached estimates and decoder scaling before choosing a small, reversible edit.

```python
import inspect

print("=== CACHED OBJECTS ===")
for name, value in sorted(list(globals().items())):
    if any(s in name.lower() for s in ("meta", "state_files", "decoder", "rd_result", "dynamic_fit")):
        if isinstance(value, dict):
            print(name, "dict keys", list(value.keys()))
        elif hasattr(value, "columns"):
            print(name, "table", value.shape, "columns", list(value.columns))
        elif hasattr(value, "shape"):
            print(name, type(value).__name__, value.shape)
        else:
            print(name, type(value).__name__, str(value)[:250])

print("\n=== DOSE DECODER CONTENTS ===")
for drug, obj0 in dose_decoders.items():
    print(drug, type(obj0).__name__)
    if isinstance(obj0, dict):
        for key, val in obj0.items():
            if hasattr(val, "shape"):
                print(" ", key, val.shape, "range", float(np.min(val)), float(np.max(val)))
            elif hasattr(val, "coef_"):
                print(" ", key, type(val).__name__, "coef", val.coef_.shape, "intercept", val.intercept_)
            else:
                print(" ", key, str(val)[:350])
    elif hasattr(obj0, "coef_"):
        print("coef", obj0.coef_.shape, "intercept", obj0.intercept_)

print("\n=== KEY FUNCTION SIGNATURES ===")
for name in ("cpu_head_parts", "cpu_head_probs", "make_head_features",
             "head_dose_data", "make_policy_data", "causal_shift_table"):
    fn0 = globals()[name]
    print(name, inspect.signature(fn0), "locals:", fn0.__code__.co_varnames)

print("\n=== PANEL AND NORMALIZATION ===")
print("hour_grid", hour_grid)
print("event_col", event_col)
print("norms", norms)
print("head feature scales:", float(np.min(hidden_scale)), float(np.median(hidden_scale)),
      float(np.max(hidden_scale)))

notes = open("/work/notes.md").read()
for title in ("## Sensor-contrast local projections", "## Dynamic identification",
              "## Recovery and threshold identification"):
    start = notes.find(title)
    if start >= 0:
        end = notes.find("\n## ", start + 4)
        section = notes[start:end if end >= 0 else len(notes)]
        print("\n" + section[:1500])
```