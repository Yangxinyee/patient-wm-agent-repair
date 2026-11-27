The recovered notes support a glucose-lowering insulin effect and a MAP-raising norepinephrine effect, subject to the identification assumptions. The architecture also gives us a separable action pathway to test. I’ll recover the current edit’s calibration details; unequal transport of doubled and halved doses would explain the rollout asymmetry.

```python
import inspect

_notes = Path("/work/notes.md").read_text()
print("Notebook characters:", len(_notes))
print("\n=== IDENTIFICATION AND EDIT DETAILS ===")
print(_notes[6500:11500])

print("\n=== PERSISTENT OBJECT NAMES ===")
_names = []
for _k, _v in list(globals().items()):
    if _k.startswith("_") or _k in {"In", "Out", "NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(_v, (np.ndarray, torch.Tensor, pd.DataFrame)):
        _names.append(f"{_k}: {type(_v).__name__}{tuple(_v.shape)}")
    elif isinstance(_v, dict):
        _names.append(f"{_k}: dict({len(_v)}), keys={list(_v)[:5]}")
    elif callable(_v):
        _names.append(f"{_k}: callable")
    elif isinstance(_v, (list, tuple)):
        _names.append(f"{_k}: {type(_v).__name__}({len(_v)})")
    elif isinstance(_v, (str, int, float, bool)):
        _names.append(f"{_k}: {repr(_v)[:70]}")
print("\n".join(_names)[:3200])

print("\n=== CURRENT VALUE-HEAD ROW EDITS ===")
for _name in ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.2.weight"]:
    _a = np.asarray(load(_name))
    _b = np.asarray(load_original(_name))
    _d = np.linalg.norm(_a - _b, axis=1)
    _ix = np.argsort(_d)[-12:][::-1]
    print(_name, [(int(i), round(float(_d[i]), 5)) for i in _ix])
print("\n=== VOCAB TOP-LEVEL KEYS ===")
print(list(VOCAB))
```