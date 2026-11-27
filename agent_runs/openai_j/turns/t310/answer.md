Norepinephrine rates are capped at 1.0 in the records, so many doubled doses are outside training support. That could contribute to the weak response, but it is not proof. Plain numeric scaling already failed. I’ll inspect a bin-compensated dose path that could preserve factual embeddings while making extrapolation smoother; it still needs rollout validation.

```python
import json, inspect, dis
import numpy as np

print("NUMERIC NOTE:", VOCAB.get("numeric_note"))
print("CODES:", VOCAB["codes"])
numeric270 = VOCAB["numeric"]
print("NUMERIC STRUCTURE:",
      list(numeric270.keys()) if isinstance(numeric270, dict) else type(numeric270))
txt270 = json.dumps(numeric270, indent=1)
print("NUMERIC TABLES:",
      txt270 if len(txt270) < 4400 else
      txt270[:900] + "\n... omitted middle ...\n" + txt270[-3400:])

print("\nACTION RECONSTRUCTION HELPERS:")
for fn_name in ("repair_atom_vectors", "repair_changed_action", "repair_core_rows"):
    fn = globals()[fn_name]
    print("\n", fn_name,
          [(p.name, p.kind.name) for p in inspect.signature(fn).parameters.values()])
    print("Referenced globals:", fn.__code__.co_names)
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = dis.Bytecode(fn).dis()
    print(src[:1250])

print("\nACTION-CACHE VARIABLES:")
cache270 = []
for name, obj in list(globals().items()):
    if name.startswith("_") or not any(s in name.lower() for s in
        ("action", "atom", "actdata", "mc145", "meta145")):
        continue
    if isinstance(obj, np.ndarray):
        cache270.append((name, str(obj.shape)))
    elif isinstance(obj, dict):
        cache270.append((name, "dict keys " + str(list(obj.keys())[:8])))
    elif hasattr(obj, "shape") and not inspect.isfunction(obj):
        cache270.append((name, str(obj.shape)))
for row in sorted(cache270)[:28]:
    print(row)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 270: off-support dose-path feasibility\n"
            "- Training NE rates have a hard maximum of 1.0; doubling often exceeds observed support.\n"
            "- This is a possible explanation for under-response, not identification of extrapolated effects.\n"
            "- Plain numeric gain previously failed. Inspecting a compensated embedding path that minimizes factual perturbations.\n"
            "- No weights changed; the calibrated gain-8.5 candidate remains installed.\n")
```