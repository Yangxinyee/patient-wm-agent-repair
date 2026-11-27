The architecture confirms that dose corrections can enter the shared recurrent state through the posterior latent, rather than through a sensor-specific head. The needed state arrays and recurrent helpers survived the reset. I’ll inspect their contracts and the earlier validation notes; an exact one-step reconstruction would make prediction-response calibration trustworthy.

```python
import inspect
from pathlib import Path

print("=== EARLIER VALIDATION AND EFFECT NOTES ===")
_notes = Path("/work/notes.md").read_text()
for _needle in ["## Python 55", "## Python 56", "## Python 57",
                "## Python 58", "## Python 59", "## Python 60",
                "## Python 135", "## Python 136"]:
    _pos = _notes.find(_needle)
    if _pos >= 0:
        _end = _notes.find("\n## ", _pos + 4)
        print(_notes[_pos:_end if _end >= 0 else _pos+1500][:1500])

print("=== RECURRENT AND HEAD TENSORS ===")
for _name in NAMES:
    if _name.startswith(("core.", "attr_heads.value", "last_", "type_emb",
                         "query_norm", "z_head", "z_post", "z_prior", "gap")):
        print(_name, SHAPES[_name])

print("=== HELPER CONTRACTS ===")
for _name in ["attention136", "block136", "core_step136", "posterior137",
              "atom138", "pool137", "mlp136", "norm136"]:
    _fn = globals()[_name]
    print("\n", _name, inspect.signature(_fn))
    try:
        print(inspect.getsource(_fn)[:3500])
    except (OSError, TypeError):
        print("names:", _fn.__code__.co_names)
        print("constants:", [x for x in _fn.__code__.co_consts
                              if isinstance(x, (str, int, float, tuple))])
        print("locals:", _fn.__code__.co_varnames)

print("=== VOCAB AND STATE CONTRACTS ===")
print("vocab keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("numeric:", numeric)
print("event kinds:", event_kind_ids)
print("bundle examples:", repr(bundles[:2])[:1200])
print("previous links:", previous[:12], "hours:", hours[:12])
print("state files:", [str(p) for p in Path("/work").glob("fwd_*/states*.npz")])
print("Active edits:", edited())
```
