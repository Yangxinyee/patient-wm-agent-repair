The early record analysis shows why raw dose–reading correlations are misleading: adjustment and dose-threshold comparisons give different effects. The recovery output was truncated, so I’ll retrieve just the validation IDs and later effect estimates. Those estimates must support the correction before its rollout predictions can count as an improvement.

```python
import re
from pathlib import Path

print("VALIDATION STAYS:", validation_stays152)
print("\n=== EFFECT ESTIMATES AND TARGETS ===")
effect_sections153 = [
    s for s in sections152
    if any(k in s.lower() for k in
           ["schedule effect", "target readout", "target shift", "true effect",
            "causal target", "record-derived", "persistence=", "beta="])
]
for s153 in effect_sections153[:9]:
    print(s153[:650])

print("\n=== FEASIBLE BASELINE AND RECENT GPU CHECKS ===")
for s153 in sections152:
    header153 = s153.splitlines()[0] if s153 else ""
    if re.search(r"(Run 77:|Run 103:|Forward 1[1-5]\b)", header153):
        print(s153[:850])

print("\n=== LIVE TARGET OBJECTS ===")
for nm153 in sorted(list(globals())):
    if re.search(r"(target|truth|rho|beta)", nm153, re.I):
        ob153 = globals()[nm153]
        if isinstance(ob153, (int, float, str)):
            print(nm153, str(ob153)[:180])
        elif isinstance(ob153, dict):
            print(nm153, "keys:", list(ob153.keys())[:10])
        elif hasattr(ob153, "shape"):
            print(nm153, "shape:", ob153.shape)

with open("/work/notes.md", "a") as f153:
    f153.write("\n## Run 153: compact validation and causal-target recovery\n"
               "Retrieved exact fixed tuning IDs and selected effect/baseline evidence after prior output truncation. "
               "No weights changed. Python runs: 153; forwards: 16.\n")
```