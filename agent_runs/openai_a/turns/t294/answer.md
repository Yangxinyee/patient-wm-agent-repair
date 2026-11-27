The records support an insulin target of roughly −10.7 for doubling and +5.35 for halving on the first 100 tuning stays. The norepinephrine estimate is still observational, so I should not treat its sign as established. I’ll inspect its policy data and the graft’s exact weight layout; a usable policy discontinuity would provide stronger evidence than regression alone.

```python
print("=== COMPLETE SUSTAINED-EFFECT NOTE ===")
for section in note_sections:
    if section.startswith("## Sustained insulin target"):
        print(section[:3600])

print("\n=== HEAD WEIGHT LAYOUT ===")
for name in validated_gated8_checkpoint:
    a = as_np_restore(validated_gated8_checkpoint[name])
    original = as_np_restore(load_original(name))
    print(name, a.shape, "changed elements", int(np.count_nonzero(a != original)))

bias_name = "attr_heads.value_bin.0.bias"
threshold_difference = (
    as_np_restore(gate035_checkpoint[bias_name])
    - as_np_restore(validated_gated8_checkpoint[bias_name])
)
activity_graft_rows = np.flatnonzero(threshold_difference)
print("Activity graft rows:", activity_graft_rows)
print("Threshold-trial bias changes:", threshold_difference[activity_graft_rows])

print("\n=== SELECTIVE-GRAFT IMPLEMENTATION ===")
try:
    selective_source_recovery = inspect.getsource(make_selective_contrast_graft)
    print(selective_source_recovery[:2800])
except Exception:
    c = make_selective_contrast_graft.__code__
    print("Global names:", c.co_names)
    print("Constants:", repr(c.co_consts)[:1000])
    import dis
    print(dis.Bytecode(make_selective_contrast_graft).dis()[-2100:])

print("\n=== PANEL STRUCTURE ===")
print(type(panel).__name__, getattr(panel, "shape", None))
if hasattr(panel, "columns"):
    print("Columns:", list(panel.columns))
    print(panel.head(3).to_string())
elif hasattr(panel, "dtype"):
    print("dtype:", panel.dtype, "first rows:", repr(panel[:2])[:900])

print("\n=== NOREPINEPHRINE ANALYSIS VARIABLES ===")
for name in sorted(list(globals())):
    if not re.match(r"^(ne_|norepi|map_|pressure)", name):
        continue
    obj = globals()[name]
    if callable(obj):
        continue
    if hasattr(obj, "shape"):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict):
        print(name, "dict keys", list(obj)[:12])
    else:
        print(name, repr(obj)[:140])
```