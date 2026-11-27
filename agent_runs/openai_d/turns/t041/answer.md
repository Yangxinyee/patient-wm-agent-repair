The architecture confirms that both the recurrent state and the last-reading register affect predictions. The packing check is therefore useful, but not yet enough to justify an edit. I’ll inspect the existing reconstruction and dose-innovation caches next; the key test is whether they expose a drug-sensitive direction that generalizes across stays.

```python
import inspect

_r35_g = list(globals().items())
_r35_lines = []
for _n, _o in sorted(_r35_g):
    if _n.startswith("_"):
        continue
    if isinstance(_o, pd.DataFrame):
        _r35_lines.append(f"{_n}: DataFrame {_o.shape}")
    elif isinstance(_o, (np.ndarray, torch.Tensor)):
        _r35_lines.append(f"{_n}: {type(_o).__name__} {tuple(_o.shape)}")
    elif isinstance(_o, dict) and _n not in ("SHAPES", "VOCAB"):
        _r35_lines.append(f"{_n}: dict keys {list(_o)[:14]}")
    elif callable(_o) and hasattr(_o, "__code__"):
        _r35_lines.append(
            f"{_n}: function args {_o.__code__.co_varnames[:_o.__code__.co_argcount]}"
        )
print("=== CACHED OBJECTS ===")
print("\n".join(_r35_lines)[:7200])

print("=== RELEVANT WEIGHT SHAPES ===")
for _n in NAMES:
    if any(_s in _n for _s in (
        "attr_heads.value_bin", "last_numeric", "last_status",
        "type_emb", "query_norm", "z_head", "z_prior", "gap"
    )):
        print(_n, SHAPES[_n])

print("=== FORWARD FILES ===")
for _p in sorted(Path("/work").glob("fwd_*/*")):
    print(str(_p))

print("=== CURRENT EDITS ===", edited())
```