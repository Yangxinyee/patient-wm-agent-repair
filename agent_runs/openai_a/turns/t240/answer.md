The failed run did not change any weights: all four tensors exactly match the validated checkpoint. The insulin target remains about −11 for doubling, supported by dose-policy threshold checks. I’ll inspect the screening function’s output shape, then use it to compare stronger candidates without repeating the scalar-conversion error.

```python
import dis

print("=== CANDIDATE AND CACHE LAYOUT ===")
for key in [(0.1, 1.0), (0.3, 1.0), (1.0, 1.0)]:
    obj = r3_candidates.get(key)
    if obj is not None:
        print(key, {
            k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:100])
            for k, v in obj.items()
        })
print("Candidate keys:", list(r3_candidates.keys()))
for label, obj in [("factual", fact_c["first"]), ("synthetic", syn_c[2.0])]:
    print(label, {
        k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:100])
        for k, v in obj.items()
    })

print("=== COMPACT EDIT BYTECODE ===")
for name in ["cache_losses", "make_selective_contrast_graft", "make_contrast_graft"]:
    print("\n" + name)
    for inst in dis.get_instructions(globals()[name]):
        if inst.opname not in {"RESUME", "CACHE", "NOP", "PRECALL"}:
            print(inst.opname, inst.argrepr)

print("=== EXISTING SCREEN CONFIGURATION ===")
for name in ["alt_contrast_config", "sc_names", "sc_nodes", "sc_rank",
             "selective_trial_lam", "selective_trial_scale"]:
    obj = globals().get(name)
    print(name, tuple(obj.shape) if hasattr(obj, "shape") else repr(obj)[:240])
```
