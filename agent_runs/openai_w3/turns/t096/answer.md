The retained estimates point to persistent insulin-driven glucose reductions and shorter positive norepinephrine effects; the original rollouts are much weaker and give the wrong sign for doubled norepinephrine. I’ll recover the local prelude replay and test how its features respond to changed doses. A useful correction feature should respond consistently to dose changes but remain small on factual inputs.

```python
import inspect

print("REPLAY FUNCTION DEPENDENCIES")
for name in ["prelude81", "state_update82", "memory82", "layer82",
             "r66_weight", "r66_prelude", "fn76", "fn77", "linear", "mlp"]:
    f = globals()[name]
    print(name, "globals:", f.__code__.co_names, "constants:",
          [x for x in f.__code__.co_consts if isinstance(x, (str, float, int))][:35])
    try:
        src = inspect.getsource(f)
        print(src[:2400])
    except (OSError, TypeError):
        pass

print("\nRECENT VARIABLES")
for k in sorted(list(globals())):
    if k.endswith(("81", "82")) or k.startswith(("r67", "sensor", "repair_", "policy", "innovation")):
        v = globals()[k]
        if callable(v):
            continue
        print(k, type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else
              list(v)[:10] if isinstance(v, dict) else str(v)[:120])

print("\nDATA/STATE GLOBALS REFERENCED BY REPLAY")
deps = set()
for name in ["prelude81", "state_update82", "memory82", "fn76", "fn77"]:
    deps.update(globals()[name].__code__.co_names)
for k in sorted(deps):
    if k not in globals():
        continue
    v = globals()[k]
    if callable(v):
        continue
    print(k, type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else
          list(v)[:12] if isinstance(v, dict) else str(v)[:140])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 85; forwards 7. Record-only reference: insulin immediate effect about -1.94/unit, persistent kernel; NE about +3.2/rate, shorter kernel. Original rollout shifts -.422,+.213,-.742,-.238 versus estimated -12.96,+6.48,+4.28,-2.14. Recovering replay dependencies for a prelude correction small on factual features and sensitive to dose contrasts.\n")
```