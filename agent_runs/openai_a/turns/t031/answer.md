The architecture confirms that doses affect the recurrent state, while reading predictions also use a separate history register. The reconstruction error could come from that register, time encoding, or latent averaging. I’ll inspect the retained code and data structures next; correcting those inputs should close the GPU mismatch without changing any weights.

```python
import inspect
from pathlib import Path

print("=== FUNCTIONS ===")
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_") or not inspect.isfunction(_obj):
        continue
    try:
        _sig = str(inspect.signature(_obj))
    except Exception:
        _sig = "?"
    print(_name + _sig)

print("\n=== ARRAYS / DATAFRAMES ===")
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_") or not hasattr(_obj, "shape"):
        continue
    print(_name, type(_obj).__name__, tuple(_obj.shape),
          str(getattr(_obj, "dtype", "")))

print("\n=== DICTIONARIES / LISTS ===")
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_") or _name in {"SHAPES", "NAMES", "VOCAB"}:
        continue
    if isinstance(_obj, dict):
        print(_name, "dict", len(_obj), "keys", str(list(_obj)[:10])[:180])
    elif isinstance(_obj, (list, tuple)):
        print(_name, type(_obj).__name__, len(_obj),
              str(_obj[:2])[:100])

print("\n=== VOCAB ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
for _key in VOCAB:
    if any(_s in _key for _s in ["norm", "categor", "unit", "status", "kind"]):
        print(_key, str(VOCAB[_key])[:1000])

print("\n=== FORWARD FILES ===")
for _path in sorted(Path("/work").glob("fwd_*/*")):
    print(_path.name, _path.stat().st_size)

print("\n=== RELEVANT TENSORS ===")
for _name in NAMES:
    if any(_s in _name for _s in [
        "last_", "attr_heads.value", "embed_transition.value", "query_norm",
        "type_emb", "z_head", "z_prior", "gap", "embed.numeric",
        "embed.bins", "out_norm", "encode_action"
    ]):
        print(_name, SHAPES[_name])
```