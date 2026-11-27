The corrected analyses consistently support a positive norepinephrine effect, but its persistence remains uncertain. The decoder has weak dose-edit sensitivity relative to its factual variation. I’ll inspect the fitting matrices to test whether a decoder aimed at minimal factual disturbance can do better. Lower factual variation at equal dose sensitivity would support that approach.

```python
import pandas as pd

diag170 = pd.DataFrame(reports147)
diag170["ne_x2_gain_per_rms"] = diag170.ne_x2_gain / diag170.ne_rms
diag170["ne_half_gain_per_rms"] = diag170.ne_half_gain / diag170.ne_rms
print("=== ALL EXISTING DECODER TRADE-OFFS ===")
print(diag170.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\n=== FITTING MATRICES ===")
for name170 in [
    "D147", "Y147", "Hfact147", "Hprobe147", "C147",
    "reg147", "B147", "HC147", "constrained147"
]:
    obj170 = globals().get(name170)
    print(name170, type(obj170).__name__, getattr(obj170, "shape", None))
    if isinstance(obj170, np.ndarray):
        print("  norm", float(np.linalg.norm(obj170)),
              "range", (float(obj170.min()), float(obj170.max())))
    elif callable(obj170):
        print(" ", inspect.signature(obj170), obj170.__code__.co_names)
    else:
        print(" ", repr(obj170)[:250])

for name170 in ["Dparts147", "Yparts147"]:
    obj170 = globals().get(name170)
    if isinstance(obj170, (list, tuple)):
        print(name170, [(getattr(v170, "shape", None),
                         float(np.linalg.norm(v170))
                         if isinstance(v170, np.ndarray) else type(v170).__name__)
                        for v170 in obj170])

print("\n=== PROBE AND INITIAL-CONSTRAINT DATA ===")
for name170 in ["prelude_cf_E111", "residual_targets102", "C146"]:
    obj170 = globals().get(name170)
    if isinstance(obj170, dict):
        print(name170, {k170: getattr(v170, "shape", type(v170).__name__)
                        for k170, v170 in obj170.items()})
    else:
        print(name170, type(obj170).__name__, getattr(obj170, "shape", None))

print("\nValidated decoder diagnostics:")
print(report_decoder147(validated_W151))
```
