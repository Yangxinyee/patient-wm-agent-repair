The earlier dose probe changed numbers but left their bins unchanged, so it did not fully represent the evaluation’s dose intervention. I’ll reconstruct the drug encoder and verify it against saved posterior states. Accurate reconstruction would let me design a targeted edit; a mismatch would need resolving first.

```python
print("VOCAB CODES:", VOCAB["codes"])
print("\nDRUG ROW EXAMPLES:")
print(train[train.event.isin(["insulin", "norepinephrine"])].head(8).to_string(index=False))
print("\nSTATE CONTAINERS:")
for name in ["S_reference", "S_probe", "repair_head_cfg", "head_cfg", "layout", "fields", "action_centers"]:
    obj1 = globals().get(name)
    if isinstance(obj1, dict):
        print(name, {k: (str(v.shape) if hasattr(v, "shape") else repr(v)[:160]) for k, v in obj1.items()})
    else:
        print(name, type(obj1).__name__)
print("\nCORE / POSTERIOR / GAP SHAPES:")
for n in NAMES:
    if any(s in n for s in ("z_post", "z_in", "core.slots", "core.out_norm", "gap.", "core.c_out", "core.a_in", "core.delta_in", "core.flow")):
        print(n, SHAPES[n])
print("\nFUNCTION DETAILS:")
for name in ["act_fn", "head_mlp", "head_norm", "hw"]:
    f1 = globals()[name]
    print(name, "names", f1.__code__.co_names, "constants", f1.__code__.co_consts)
print("\nTRAINING STATE METADATA:")
print(state_meta.head(12).to_string(index=False))
print("\nDOSE POLICY SUMMARY:")
print(policy.to_string(index=False)[:1800])
```
