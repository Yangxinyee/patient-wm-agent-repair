The validated checkpoint is restored. The architecture places the repair before several normalizations and recurrent layers, so a large weight change need not produce a proportional outcome change. I’ll inspect the installed transport and its replay helpers. Saturation or cancellation would explain the failed gain adjustment; approximately linear responses would refute that explanation.

```python
import inspect
import types

notes245 = Path("/work/notes.md").read_text()
print("EARLY RECORD IDENTIFICATION\n", notes245[:8000])
for marker245 in ("Python run 193", "Python run 206", "Python run 220"):
    p245 = notes245.find(marker245)
    if p245 >= 0:
        print("\nNOTE", marker245, "\n", notes245[max(0, p245-500):p245+1800])

print("\nMODEL EDIT SCALES")
for n245 in edited():
    a245 = np.asarray(load(n245))
    o245 = np.asarray(load_original(n245))
    print(n245, tuple(a245.shape), "original norm", round(float(np.linalg.norm(o245)), 3),
          "edit norm", round(float(np.linalg.norm(a245-o245)), 3),
          "max delta", round(float(np.max(np.abs(a245-o245))), 3))
print("dd193 column norms", np.linalg.norm(dd193, axis=0))

print("\nLOCAL FUNCTION INVENTORY")
fnames245 = sorted(k for k, v in list(globals().items())
                   if isinstance(v, types.FunctionType) and v.__module__ == "__main__")
for k245 in fnames245:
    f245 = globals()[k245]
    print(k245, f245.__code__.co_varnames[:f245.__code__.co_argcount])

print("\nTRANSPORT-RELATED RETAINED OBJECTS")
for k245 in sorted(list(globals())):
    if not any(s245 in k245 for s245 in ("193", "206", "220", "185")):
        continue
    v245 = globals()[k245]
    if hasattr(v245, "shape") and (getattr(v245, "ndim", 0) <= 1 or 192 in tuple(v245.shape)):
        print(k245, type(v245).__name__, tuple(v245.shape))
    elif isinstance(v245, (int, float, str)) and len(str(v245)) < 150:
        print(k245, repr(v245))

print("\nHELPER IMPLEMENTATIONS")
for k245 in ("prelude155", "preludeE185", "prelude_parts145", "prelude_cache88"):
    f245 = globals()[k245]
    try:
        print(k245, inspect.getsource(f245))
    except (OSError, TypeError):
        print(k245, "global references", f245.__code__.co_names,
              "defaults", f245.__defaults__, "constants", f245.__code__.co_consts[:15])

with open("/work/notes.md", "a") as f245:
    f245.write("\nPython run 245; forwards 18. Restored validated checkpoint222 via checkpoint243. "
               "Inspecting transport decomposition and normalization scales after rank-two adjustment had little rollout effect. "
               "No new weight changes.\n")
```