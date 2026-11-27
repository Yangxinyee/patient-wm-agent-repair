The recovered analysis contains negative insulin estimates from sensor-based and dosing-threshold checks, but also warns that several alternative estimators are unreliable. Norepinephrine identification remains unresolved. The output cut off before the edit-design summaries, so I’ll retrieve those compactly. Stable threshold estimates would support the insulin correction; instability would weaken confidence in its magnitude.

```python
import json

print("=== COMPACT EVIDENCE AND DESIGN SUMMARIES ===")
for i in [12, 13, 57, 65, 81, 93, 98]:
    lines = _sections295[i].splitlines()
    summary = "\n".join(
        [lines[0]] + [line for line in lines[1:] if line.startswith("- ")]
    )
    print(summary[:750], "\n")

print("=== INSULIN THRESHOLD ESTIMATES ===")
_s297 = _sections295[12]
_start297 = _s297.find("{")
if _start297 >= 0:
    try:
        _j297, _ = json.JSONDecoder().raw_decode(_s297[_start297:])
        _rd297 = _j297.get("rd_checks", {})
        for key, result in list(_rd297.items())[:16]:
            print(
                key,
                "beta", round(float(result["beta"]), 4),
                "SE", round(float(result["se"]), 4),
                "dose_jump", round(float(result["first_stage_jump"]), 4),
                "n", result["n"]
            )
    except (ValueError, KeyError, TypeError) as exc:
        print("Compact JSON extraction failed:", type(exc).__name__)

print("\n=== RECORD-DERIVED INSULIN DYNAMIC MODEL ===")
print("beta:", globals().get("_target_beta"))
print("persistence:", globals().get("_target_rho"))
if "ins_target" in globals():
    print("Training-record insulin_x2 target:", float(ins_target[0]))
    print("Target contributing stay-hours:", ins_target[1])
print("\nNo weights changed; current edit remains amplitude093_checkpoint.")
```
