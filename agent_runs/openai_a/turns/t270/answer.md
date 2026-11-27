The notebook supports a negative insulin effect near −1.93 per unit-hour, with persistence implying roughly −11 for doubling and +5.5 for halving. Norepinephrine is not yet causally identified. I will first inspect how the insulin correction is encoded; an activity gate should reduce factual cost without discarding the identified insulin response.

```python
import dis

for name, limit in [
    ("make_selective_contrast_graft", 3400),
    ("make_contrast_graft", 1800),
    ("selective_cache", 1400),
    ("screen_value_candidate", 1200),
]:
    func_obj = globals()[name]
    print("\n=== " + name + " ===")
    try:
        text = inspect.getsource(func_obj)
    except Exception:
        text = dis.Bytecode(func_obj).dis()
    print(text[:limit])
    print("GLOBAL REFERENCES:", func_obj.__code__.co_names)

print("\n=== FIT AND CACHE STRUCTURES ===")
for name in ["_activity_best", "r3_candidates", "protected_separators",
             "sc_fact", "sc_syn", "sc_active", "stronger6_checkpoint",
             "scale7_checkpoint"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, "keys:", list(obj.keys()))
        if name in {"sc_fact", "sc_syn", "r3_candidates"} and obj:
            k = next(iter(obj))
            v = obj[k]
            if isinstance(v, dict):
                print(" first entry:", {
                    kk: (tuple(vv.shape) if hasattr(vv, "shape") else
                         repr(vv)[:100])
                    for kk, vv in v.items()
                })
    else:
        print(name, type(obj).__name__)
```
