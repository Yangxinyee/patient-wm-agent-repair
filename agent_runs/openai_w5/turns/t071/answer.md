Dose information is present at drug bundles: held-out rate decoding reached about 0.94 R² for insulin and 0.76 for norepinephrine. Combining context with the action vector also decoded unexpected doses better than the existing posterior hidden layer. I’ll investigate two explicit dose-innovation channels in that layer, first checking how latent changes affect the next reading. Stable, drug-specific sensitivity would support this approach; weak or strongly cross-coupled sensitivity would refute it.

```python
import inspect, dis, numpy as np, torch

def short_impl60(f, limit):
    try:
        text = inspect.getsource(f)
    except Exception:
        text = " ; ".join(
            f"{i.opname} {i.argrepr}" for i in dis.get_instructions(f)
            if i.opname not in {"RESUME", "CACHE", "NOP"})
    return text[-limit:]

print("=== RECURRENCE RETURN / END ===")
print(short_impl60(wm_step_saved, 3500))
print("\n=== CONTEXT HELPER ===")
print(short_impl60(context_from_saved, 2500))
print("\n=== HEAD INPUT DEPENDENCIES ===")
print("reconstruct_head globals:", reconstruct_head.__code__.co_names)
print("head_features_layout globals:", head_features_layout.__code__.co_names)
print(short_impl60(head_features_layout, 1600))

print("\n=== TRAIN READING CACHE ===")
for key in ["base", "dt", "previous", "codes", "lastage", "haslast", "lastval", "lastbin"]:
    value60 = cacheTrain[key]
    print(key, type(value60).__name__, getattr(value60, "shape", None),
          "sample", as_np(value60).reshape(-1)[:5])
print(cacheTrain["df"].head(6).to_string(index=False))

print("\n=== POSTERIOR OUTPUT AND STATE CONVENTIONS ===")
for name in ["z_post.0.weight", "z_post.0.bias", "z_post.2.weight",
             "z_post.2.bias", "z_in.weight", "z_head.weight"]:
    print(name, SHAPES[name])
print("Saved-state keys/shapes:",
      {k: np.shape(v) for k, v in s4.items()})
print("Context-return shapes:",
      [getattr(v, "shape", None) for v in context_from_saved(np.array([10]), s4)])
print("Head layout:", head_layout_spec)
print("Forward folders:",
      [(p.name, [q.name for q in p.iterdir()])
       for p in sorted(Path("/work").glob("fwd_*"))])
print("Edits:", edited())
```
