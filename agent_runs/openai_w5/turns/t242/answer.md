The recovered notes estimate a glucose decrease after insulin and a MAP increase after norepinephrine, with effects persisting across hours. These remain conditional on the recorded-history adjustment assumptions. The architecture also confirms that private dose signals can influence ordinary features, so centering leakage is plausible. The output was truncated; I’ll identify the available diagnostic objects before testing that pathway.

```python
import inspect
import pandas as pd

_items215 = list(globals().items())
_groups215 = {"functions": [], "arrays": [], "tables": [], "dicts": [], "other": []}
for _n215, _v215 in _items215:
    if _n215.startswith("_") or _n215 in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if callable(_v215) and getattr(_v215, "__module__", None) == "__main__":
        try:
            _sig215 = str(inspect.signature(_v215))
        except Exception:
            _sig215 = ""
        _groups215["functions"].append(_n215 + _sig215)
    elif isinstance(_v215, (np.ndarray, torch.Tensor)):
        _groups215["arrays"].append(f"{_n215}: {tuple(_v215.shape)}")
    elif isinstance(_v215, pd.DataFrame):
        _groups215["tables"].append(
            f"{_n215}: {_v215.shape}, columns={list(_v215.columns)[:12]}"
        )
    elif isinstance(_v215, dict):
        _groups215["dicts"].append(
            f"{_n215}: {len(_v215)} keys; {list(_v215)[:6]}"
        )
    elif isinstance(_v215, (tuple, list)):
        _groups215["other"].append(f"{_n215}: {type(_v215).__name__}[{len(_v215)}]")

for _group215, _rows215 in _groups215.items():
    print("===", _group215.upper(), "===")
    print("\n".join(_rows215)[:({"functions": 2600, "arrays": 1500,
                               "tables": 900, "dicts": 2000, "other": 600}[_group215])])

print("=== NOTEBOOK SECTION TITLES ===")
print("\n".join(
    line for line in Path("/work/notes.md").read_text().splitlines()
    if line.startswith("## ")
)[:1800])
```
