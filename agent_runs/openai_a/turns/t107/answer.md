The recovered results support a negative insulin effect, but norepinephrine estimates depend strongly on adjustment choices and its sensor instrument is weak. I won’t treat those estimates as established effects. Next I’ll inspect the threshold analysis and local-step helpers; consistency across thresholds would strengthen identification.

```python
print("=== THRESHOLD AND SUSTAINED-EFFECT RESULTS ===")
for _r93_section in _r92_sections:
    if _r93_section.startswith((
        "## Recovery and threshold identification;",
        "## Sustained insulin target and NE flexible adjustment;",
        "## Dynamic identification; Python 10"
    )):
        print(_r93_section[:2300])

print("\n=== LOCAL RECONSTRUCTION VARIABLES ===")
for _r93_name, _r93_obj in list(globals().items()):
    if _r93_name.startswith(("rm_", "robust", "graft", "gate", "protected")):
        if hasattr(_r93_obj, "shape"):
            print(_r93_name, tuple(_r93_obj.shape))
        elif isinstance(_r93_obj, dict):
            print(_r93_name, "keys:", list(_r93_obj.keys())[:20])
        elif not inspect.isfunction(_r93_obj):
            print(_r93_name, repr(_r93_obj)[:160])

print("\n=== HELPER DEFINITIONS ===")
for _r93_name in ["rm_lin", "rm_ln", "rm_mlp", "rm_set", "rm_insulin_atoms"]:
    _r93_obj = globals().get(_r93_name)
    if _r93_obj is None:
        continue
    print(_r93_name, inspect.signature(_r93_obj),
          "source:", _r93_obj.__code__.co_filename)
    try:
        print(inspect.getsource(_r93_obj)[:2500])
    except Exception:
        print("Referenced names:", _r93_obj.__code__.co_names)
        print("Constants:", repr(_r93_obj.__code__.co_consts)[:700])
```
