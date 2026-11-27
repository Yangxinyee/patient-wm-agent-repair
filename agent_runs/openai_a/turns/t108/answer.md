The threshold results put insulin’s immediate effect near −1.9 per unit, with persistence around 0.83. That implies a much larger sustained effect than the current rollout predicts. I’ll inspect the encoding helpers next; accurate reconstruction is needed to build a correction that responds to dose without unstable history feedback.

```python
import json, dis

print("=== ENCODING HELPER SOURCE ===")
for _r94_name, _r94_limit in [
    ("rm_set", 2200), ("rm_insulin_atoms", 3000), ("rm_mlp", 1000)
]:
    _r94_obj = globals()[_r94_name]
    print("\n", _r94_name, inspect.signature(_r94_obj))
    try:
        _r94_source = inspect.getsource(_r94_obj)
    except Exception:
        _r94_source = dis.Bytecode(_r94_obj).dis()
    print(_r94_source[:_r94_limit])

print("\n=== RECONSTRUCTION CACHE NAMES ===")
print(", ".join(n for n in globals() if n.startswith("rm_")))

print("\n=== VOCAB LAYOUT ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
for _r94_key in VOCAB:
    if any(s in _r94_key for s in ("categor", "unit", "norm")):
        _r94_v = VOCAB[_r94_key]
        print(_r94_key, str(_r94_v)[:1600])

print("\n=== PANEL LAYOUT ===")
print("panel columns:", list(panel.columns))
print("hour_grid:", hour_grid)
print("training panel example:", panel.head(4).to_string())
print("\n=== INITIAL STATE / ENCODER TENSORS ===")
print("\n".join(f"{n}: {SHAPES[n]}" for n in NAMES
                if n.startswith(("embed.", "encode.", "encode_action.",
                                 "core.z_", "core.gap", "core.out_norm",
                                 "core.slots", "core.init", "core.flow"))))
```
