The scalar dose probe appears to be the bottleneck: stronger corrections also introduce factual errors. I’ll inspect the local state and reading-head arrays so I can constrain predicted reading changes directly. That approach is worth pursuing only if it preserves dose sensitivity while reducing factual output shifts.

```python
print("=== READING-HEAD AND STATE ARRAYS ===")
_report119 = []
for _name119, _obj119 in sorted(list(globals().items())):
    if _name119.startswith("_"):
        continue
    if (re.search(r"(45|46|82)$", _name119)
        or _name119.startswith(("core_", "hnew", "h_new", "base_h", "local_h"))
        or _name119 in ("W0", "st", "state_data", "metadata", "meta", "best_config")):
        if isinstance(_obj119, (np.ndarray, torch.Tensor)):
            _report119.append(f"{_name119}: {tuple(_obj119.shape)}, {_obj119.dtype}")
        elif isinstance(_obj119, dict):
            _report119.append(f"{_name119}: dict keys {list(_obj119)[:9]}")
        elif isinstance(_obj119, (list, tuple)):
            _report119.append(f"{_name119}: {type(_obj119).__name__} len={len(_obj119)}, first={str(_obj119[:1])[:120]}")
        elif isinstance(_obj119, (str, int, float, bool, np.number)):
            _report119.append(f"{_name119}: {str(_obj119)[:180]}")
print("\n".join(_report119)[:5000])

print("\n=== DATAFRAME METADATA ===")
import pandas as pd
print("\n".join(f"{n}: {o.shape}, columns={list(o.columns)[:9]}"
    for n, o in list(globals().items())
    if not n.startswith("_") and isinstance(o, pd.DataFrame))[:1600])

print("\n=== DECODER HELPER BYTECODE ===")
import dis, io, contextlib
_buf119 = io.StringIO()
with contextlib.redirect_stdout(_buf119):
    dis.dis(next_means45)
print(_buf119.getvalue()[:2000])
print("core_update defaults:", core_update_local.__defaults__)
print("core_update locals:", core_update_local.__code__.co_varnames)
print("currently edited:", edited())
with open("/work/notes.md", "a") as _f119:
    _f119.write("\n## Run 119: inspect output-constrained inputs\n"
                "Preparing a closed-form correction constrained on actual reading outputs, rather than scalar probe leakage alone. "
                "Inspecting retained local states, reading-head arrays and metadata. No weights changed. "
                "Python runs: 119; forwards: 15.\n")
```