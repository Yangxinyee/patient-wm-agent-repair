```python
import re, inspect
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

r214_notes = Path("/work/notes.md").read_text()
r214_sections = re.split(r"(?=^## )", r214_notes, flags=re.M)
r214_evidence = [
    s for s in r214_sections
    if re.search(r"bootstrap|identif|instrument|equal.stay|13\.2008|3\.6087|causal.*estimate|record.*target", s, re.I)
]
print("\n=== RECORD-BASED EVIDENCE ===")
print("\n".join(r214_evidence[:18])[:15000])

print("\n=== RETAINED ESTIMATORS AND CANDIDATE INTERFACES ===")
for r214_name, r214_obj in sorted(list(globals().items())):
    if r214_name.startswith("r214_"):
        continue
    if callable(r214_obj) and hasattr(r214_obj, "__code__") and re.search(
        r"dual|candidate|direction|factual|rollout|causal|target|dose_metrics|gated_ne|panel", r214_name
    ):
        print(r214_name, str(inspect.signature(r214_obj))[:250],
              "dependencies:", r214_obj.__code__.co_names)

print("\n=== EFFECT / TRANSMISSION OBJECTS ===")
for r214_name, r214_obj in sorted(list(globals().items())):
    if r214_name.startswith("r214_") or callable(r214_obj):
        continue
    if re.search(r"target|causal|effect|kernel|calibrations192|results194|result87|rate192|active192|gate_fits180", r214_name, re.I):
        if hasattr(r214_obj, "shape"):
            print(r214_name, r214_obj.shape)
        elif isinstance(r214_obj, dict):
            print(r214_name, repr(r214_obj)[:1600])
        elif isinstance(r214_obj, (float, int, str, list, tuple)):
            print(r214_name, repr(r214_obj)[:1000])

print("\nSaved edits:", edited())
with open("/work/notes.md", "a") as r214_file:
    r214_file.write(
        "\n## Run 214\n"
        "Recovered architecture, record-identification evidence and retained repair interfaces. "
        "Current GPU-passing edit has strongly asymmetric half/double rollout effects; "
        "dose transmission and generated-state feedback must be distinguished before further calibration. "
        "No saved weights changed.\n"
    )
```