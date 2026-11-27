The architecture confirms that predictions combine recurrent state, latent sampling, and a same-code reading register. The previous output was truncated before the saved objects appeared. I will inspect those separately so we can distinguish an implementation error from sampling noise; a correct reconstruction should match both means and bin probabilities.

```python
import inspect
import torch
torch.set_grad_enabled(False)

print("=== TABLES / ARRAYS ===")
for _n, _o in sorted(list(globals().items())):
    if _n.startswith("_"):
        continue
    if isinstance(_o, (pd.DataFrame, pd.Series, np.ndarray)):
        print(_n, type(_o).__name__, _o.shape)

print("\n=== DICTIONARIES ===")
for _n, _o in sorted(list(globals().items())):
    if _n.startswith("_") or not isinstance(_o, dict):
        continue
    print(_n, list(_o)[:18])

print("\n=== ANALYSIS FUNCTIONS ===")
for _n, _o in sorted(list(globals().items())):
    if callable(_o) and getattr(_o, "__module__", "") == "__main__":
        try:
            print(_n, str(inspect.signature(_o)))
        except Exception:
            print(_n)

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
for _k in ("config", "codes", "normalisation", "normalization"):
    if _k in VOCAB:
        print(_k, VOCAB[_k])

print("\n=== HEAD / REGISTER / PRIOR TENSORS ===")
for _n in NAMES:
    if any(_s in _n for _s in (
        "attr_heads", "last_", "transition", "query_", "type_emb",
        "z_prior", "z_head", "out_norm", "core.gap", "core.flow"
    )):
        print(_n, SHAPES[_n])

print("\n=== SAVED STATE FILE ===")
with np.load("/work/fwd_3/states_original.npz") as _s:
    for _k in _s.files:
        _v = _s[_k]
        print(_k, _v.shape, _v.dtype, "first:", _v[:2] if _v.ndim == 1 else "")
print("\n=== FACTUAL OUTPUT COLUMNS ===")
for _p in sorted(Path("/work").glob("fwd_*/factual_original.csv")):
    _d = pd.read_csv(_p)
    print(str(_p), _d.shape, list(_d.columns))
    print(_d.head(2).to_string(index=False))
```
