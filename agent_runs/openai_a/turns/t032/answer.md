The inspection output was truncated because a function signature printed its entire weight dictionary; the kernel itself did not fail. I’ll repeat it with bounded summaries so we can locate the reconstruction inputs. This is still diagnostic work—no weights have changed.

```python
import inspect
from pathlib import Path

_lines = []
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_") or inspect.ismodule(_obj):
        continue
    if inspect.isfunction(_obj):
        _c = _obj.__code__
        _lines.append(f"{_name}: function args={_c.co_varnames[:_c.co_argcount]}")
    elif hasattr(_obj, "shape"):
        _lines.append(f"{_name}: {type(_obj).__name__} shape={tuple(_obj.shape)}")
    elif isinstance(_obj, dict):
        _lines.append(f"{_name}: dict[{len(_obj)}]")
    elif isinstance(_obj, (list, tuple)):
        _lines.append(f"{_name}: {type(_obj).__name__}[{len(_obj)}]")
print("=== COMPACT INVENTORY ===")
print("\n".join(_lines)[:4800])

print("\n=== VOCAB ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
for _key in ["normalization", "normalisation", "categorical", "units"]:
    if _key in VOCAB:
        print(_key, str(VOCAB[_key])[:700])

print("\n=== FILES ===")
for _path in sorted(Path("/work").glob("fwd_*/*")):
    print(str(_path), _path.stat().st_size)

print("\n=== HEAD TENSORS ===")
for _name in NAMES:
    if any(_s in _name for _s in [
        "last_", "attr_heads.value", "embed_transition.value", "query_norm",
        "type_emb", "z_head", "z_prior", ".gap"
    ]):
        print(_name, SHAPES[_name])

print("\n=== REMAINING EARLY NOTEBOOK HEADINGS ===")
for _line in Path("/work/notes.md").read_text().splitlines():
    if _line.startswith("##"):
        print(_line)
```