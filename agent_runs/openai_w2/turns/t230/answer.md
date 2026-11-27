The error is resolved: bundles without actions use the encoded empty action set, not a zero vector. With that convention, posterior RMS error falls to about 0.000036. I’ll rebuild the prelude features and inspect the existing decoder fit so these independent training states can help reduce factual disturbance.

```python
r198_u = r197_variants["encoded_empty_when_no_action"]
r198_prelude = prelude_features111(hf188, e191, r198_u, z188)
print("PRELUDE RETURN LAYOUT")
if isinstance(r198_prelude, (tuple, list)):
    print([(type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else None)
           for v in r198_prelude])
    r198_matches = [
        v for v in r198_prelude
        if hasattr(v, "shape") and tuple(v.shape) == (len(rows188), 4, 192)
    ]
else:
    print(type(r198_prelude).__name__, tuple(r198_prelude.shape))
    r198_matches = [r198_prelude]
print("old prelude return layout:", [
    tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
    for v in prelude_ret189
])

print("\nDECODER OBJECTS")
for r198_name in (
    "validated_W151", "validated_W176", "validated_gains151",
    "validated_B151", "validated_B176", "candidate173",
    "constraint_candidates147"
):
    r198_obj = globals().get(r198_name)
    if isinstance(r198_obj, dict):
        print(r198_name, "keys", list(r198_obj)[:15])
        if r198_name == "constraint_candidates147":
            r198_item = r198_obj[(0.3, 0.0001)]
            print("selected entry type", type(r198_item).__name__)
            if isinstance(r198_item, dict):
                print({k: (tuple(v.shape) if hasattr(v, "shape") else str(v)[:100])
                       for k, v in r198_item.items()})
            else:
                print("selected entry", getattr(r198_item, "shape", None),
                      str(r198_item)[:250])
    elif r198_obj is not None:
        print(r198_name, getattr(r198_obj, "shape", None),
              str(r198_obj)[:160])

print("\nFIT / EDIT VARIABLES")
for r198_name, r198_obj in sorted(list(globals().items())):
    if (
        r198_name.endswith(("146", "147", "170", "171", "172", "173", "175", "176"))
        and hasattr(r198_obj, "shape")
    ):
        print(r198_name, tuple(r198_obj.shape))
print("\nFIT / EDIT FUNCTIONS")
for r198_name, r198_obj in sorted(list(globals().items())):
    if inspect.isfunction(r198_obj) and any(
        s in r198_name.lower() for s in ("fit", "candidate", "edit", "decoder", "constraint")
    ):
        print(r198_name, inspect.signature(r198_obj),
              r198_obj.__code__.co_names)

with open("/work/notes.md", "a") as r198_notes:
    r198_notes.write(
        "\n## Runs193–198: independent-state reconstruction fixed\n"
        "Cached weights and time flow were correct. The remaining error was "
        "zero action input for bundles without actions: the forward implementation "
        "uses the encoded empty action set. Correcting this gives posterior RMS "
        "3.646e-5 across 4044 noninitial bundles, with maximum per-bundle RMS "
        "1.073e-4. Drug bundles also match at ~4e-5. "
        "validated_B176 is unchanged. New training-state features can now be "
        "used after confirming the prelude return layout.\n"
    )
```
