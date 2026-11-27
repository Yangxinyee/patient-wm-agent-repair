The last result reproduced the drug posterior to about 0.000074 RMSE. That supports the infusion-input reconstruction, but does not establish the causal effects. I’ll now recover the transition details and retained causal analyses before testing an edit. If controlled dose perturbations reveal a weak action response, that supports repairing this pathway; an adequate response would point elsewhere.

```python
import json, inspect, re
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED FUNCTIONS ===")
for r40_name, r40_obj in list(globals().items()):
    if r40_name.startswith("_") or r40_name.startswith("r40_"):
        continue
    if inspect.isfunction(r40_obj) and r40_obj.__module__ == "__main__":
        try:
            print(r40_name, inspect.signature(r40_obj))
        except Exception:
            print(r40_name)

print("\n=== RELEVANT RETAINED OBJECTS ===")
r40_pattern = re.compile(
    r"act|atom|post|prior|prectx|dose|caus|head|state|panel|kernel|recipe|effect|innov|lag|truth|best",
    re.I,
)
for r40_name, r40_obj in list(globals().items()):
    if r40_name.startswith("_") or r40_name.startswith("r40_"):
        continue
    if not r40_pattern.search(r40_name) or callable(r40_obj):
        continue
    if hasattr(r40_obj, "shape"):
        print(r40_name, type(r40_obj).__name__, tuple(r40_obj.shape))
        if hasattr(r40_obj, "columns"):
            print("  columns:", list(r40_obj.columns))
    elif isinstance(r40_obj, dict):
        print(r40_name, "dict keys:", list(r40_obj)[:25])
    elif isinstance(r40_obj, (list, tuple)):
        print(r40_name, type(r40_obj).__name__, "length", len(r40_obj),
              "preview", repr(r40_obj[:3])[:250])
    elif isinstance(r40_obj, (str, int, float, bool)):
        print(r40_name, repr(r40_obj)[:300])

print("\n=== EARLIER CAUSAL NOTES ===")
r40_notes = Path("/work/notes.md").read_text()
print(r40_notes[:6500])
print("\n=== CURRENT EDITS ===", edited())

with open("/work/notes.md", "a") as r40_file:
    r40_file.write(
        "\n## Run 40 (restart)\n"
        "Recovered architecture, schema, retained reconstruction and causal objects. "
        "Run-39 isolated drug posterior reconstruction was accurate to approximately "
        "7.4e-5 RMSE after including the learned empty-measurement encoding. "
        "This validates that component, not the causal effects or a weight edit. No edits.\n"
    )
```
