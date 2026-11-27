The records support effects opposite to the raw dose–reading associations, and the architecture explains how those associations could dominate the model. I’ll recover the lag estimates and reconstruction helpers next. The proposed edit should respond to dose deviations while staying small on ordinary recorded trajectories; held-out tests must demonstrate that separation.

```python
import re

print("=== IDENTIFICATION AND LAG CHECKS ===")
print(Path("/work/notes.md").read_text()[4300:9900])

snapshot = list(globals().items())
print("\n=== LOCAL FUNCTIONS ===")
print(" ".join(sorted(
    name for name, obj in snapshot
    if callable(obj) and getattr(obj, "__module__", None) == "__main__"
)))

print("\n=== RELEVANT RETAINED OBJECTS ===")
pattern = re.compile(r"post|latent|state|cache|panel|impulse|lag|tune|tuning|exact|single|conv|feature", re.I)
for name, obj in sorted(snapshot):
    if name.startswith("_") or not pattern.search(name):
        continue
    if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys:", list(obj)[:8])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj), repr(obj[:4])[:180])

print("\n=== OTHER GLOBAL NAMES ===")
print(" ".join(sorted(name for name, obj in snapshot
                      if not name.startswith("_") and not pattern.search(name))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 38; forward 6\n"
            "- Architecture recovered. Shared recurrent updates permit dose corrections, but all-head loss must also be checked.\n"
            "- Recovering lag evidence and exact posterior helpers; no weights changed.\n")
```