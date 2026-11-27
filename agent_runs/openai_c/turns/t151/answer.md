The architecture permits compensation at the set encoders’ first linear layers, so their inputs could remain unchanged while the reading head receives a transformed query. The output was truncated before the saved directions appeared. I’ll recover those next; their orientation and calibration must be clear before this is a justified experiment.

```python
import re
import inspect
from pathlib import Path

nr107 = Path("/work/notes.md").read_text()
print("=== EARLIER HEAD EXPERIMENTS ===")
sections107 = re.split(r"(?m)(?=^## )", nr107)
for section in sections107:
    heading = section.splitlines()[0] if section.splitlines() else ""
    if re.search(r"\brun[s]?\s+7[0-5]\b", heading, flags=re.I):
        print(section[:1600])

print("\n=== RECORD TARGET / SAFE BASELINE NOTES ===")
for section in sections107:
    heading = section.splitlines()[0] if section.splitlines() else ""
    if ("gain 4" in section.lower() and
        any(s in section.lower() for s in ("nll", "validated", "safe")) and
        not re.search(r"\b(8[3-9]|9[0-9]|10[0-7])\b", heading)):
        print(section[:1100])

print("\n=== RETAINED HEAD VARIABLES ===")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    if any(s in name.lower() for s in
           ("72", "73", "74", "probe", "target", "shear", "head_basis")):
        obj = globals()[name]
        if callable(obj):
            try:
                print(name, "function", inspect.signature(obj))
            except Exception:
                print(name, "callable")
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape)
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:20])
        elif isinstance(obj, (list, tuple)):
            print(name, "length", len(obj), "sample", repr(obj[:6])[:220])
        else:
            print(name, type(obj).__name__, repr(obj)[:200])

print("\n=== COMPENSATION TARGETS ===")
for name in NAMES:
    if name.startswith("attr_heads.") or name.startswith("monitor_head.") or name.startswith("last_"):
        print(name, SHAPES[name])

print("\n=== TRANSFORM HELPER SIGNATURES ===")
for name in ("apply_lowrank_left_transform", "apply_lowrank_right_transform",
             "orthogonal_reflection_sequence", "orthogonal_rank1_factors",
             "effective_transform"):
    obj = globals().get(name)
    if callable(obj):
        print(name, inspect.signature(obj))
        try:
            print(inspect.getsource(obj)[:1500])
        except Exception:
            print("Source unavailable; names:", obj.__code__.co_varnames,
                  "constants:", obj.__code__.co_consts)

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 107: compensation feasibility\n"
            "- Architecture has a linear first layer before encoder nonlinearities.\n"
            "- A coordinated embedding/register basis change can isolate query modulation at the value head.\n"
            "- Recovering exact direction calibration and transform helpers; no edits changed.\n"
            "- Forward passes used: 40; 10 remain.\n")
```