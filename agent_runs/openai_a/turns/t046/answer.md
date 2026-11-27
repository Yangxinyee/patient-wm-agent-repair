The architecture confirms that doses affect recurrent state and that generated readings feed back during rollouts. The earlier records support a negative insulin effect, but norepinephrine estimates depend strongly on adjustment. I’ll recover the remaining identification checks and tensor names before choosing an edit; inconsistent causal estimates would argue against treating an observational association as a drug effect.

```python
import re
import inspect

print("=== NOTEBOOK SECTION INDEX ===")
_recover_notes = Path("/work/notes.md").read_text()
for _m in re.finditer(r"(?m)^## .*$", _recover_notes):
    print(_m.start(), _m.group())
print("\n=== TENSOR NAMES AND SHAPES ===")
for _n in NAMES:
    if any(s in _n for s in
           ("last_", "attr_heads", "embed_transition", "embed.numeric",
            "z_head", "z_prior", "encode_action", "query_norm", "type_emb")):
        print(_n, SHAPES[_n])
print("\n=== RETAINED VARIABLE NAMES ===")
print(" | ".join(k for k in globals() if not k.startswith("_"))[:3500])
print("\n=== CPU HEAD FUNCTION ===")
for _fn in ("cpu_head_parts", "cpu_head_probs"):
    if _fn in globals():
        print(_fn, inspect.signature(globals()[_fn]))
        try:
            print(inspect.getsource(globals()[_fn])[:2500])
        except Exception as _exc:
            print("Source unavailable:", str(_exc)[:120])
print("\n=== LAST-NUMERIC WEIGHTS ===")
for _n in NAMES:
    if "last_numeric" in _n:
        _a = load_original(_n)
        print(_n, _a.shape)
        if _a.ndim == 2 and _a.shape[1] <= 16:
            print("column norms:", np.linalg.norm(_a, axis=0))
            print("first rows:", _a[:4])
```
