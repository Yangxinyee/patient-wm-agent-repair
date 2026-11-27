The architecture has an explicit action pathway, yet the original rollouts respond very little to dose changes. The earlier threshold estimates suggest a larger insulin effect, though observation selection needs care. I will retrieve the retained analysis objects and relevant tensor shapes without another oversized printout. This will let me test a targeted action-path repair; failure to preserve factual predictions would argue against it.

```python
print("=== EARLY IDENTIFICATION NOTES, RUNS 5–6 ===")
_s = _notes.find("## Run 5")
_e = _notes.find("## Run 7")
print(_notes[_s:_e][-2200:])

print("\n=== RETAINED ARRAYS AND DATAFRAMES ===")
_items = []
for _n, _o in list(globals().items()):
    if _n.startswith("_"):
        continue
    if isinstance(_o, pd.DataFrame):
        _items.append(f"{_n}: DataFrame {_o.shape}")
    elif isinstance(_o, np.ndarray) and _o.size > 100:
        _items.append(f"{_n}: ndarray {_o.shape}")
    elif isinstance(_o, dict) and _n not in {"SHAPES", "VOCAB"}:
        _items.append(f"{_n}: dict {list(_o.keys())[:8]}")
    elif inspect.isfunction(_o) and getattr(_o, "__module__", "") == "__main__":
        _items.append(f"{_n}: function {inspect.signature(_o)}")
print("\n".join(_items)[:4200])

print("\n=== ACTION AND VALUE TENSORS ===")
for _n in NAMES:
    if (_n.startswith(("encode_action.", "embed.numeric", "core.z_post",
                       "attr_heads.value", "core.prelude.0", "last_numeric"))
        or "transition.value" in _n):
        print(_n, SHAPES[_n])

print("\n=== ORIGINAL ROLLOUT SUMMARY ===")
for _folder in sorted(os.listdir("/work")):
    if not _folder.startswith("fwd_"):
        continue
    for _file in os.listdir("/work/" + _folder):
        if _file.startswith("dose_edit_") and _file.endswith(".csv"):
            _d = pd.read_csv("/work/" + _folder + "/" + _file)
            _v = _d[_d["reached"].astype(str).str.lower().isin(["true", "1"])]
            print(_folder, _file,
                  _v.groupby("edit")["shift"].agg(["mean", "count"]).round(5).to_dict())

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 21\n"
             "Run 20 output was truncated. Retrieved compact retained-object inventory, "
             "action/value tensor shapes and original rollout effects. Explicit action pathway "
             "and same-code measurement shortcut suggest a targeted correction rather than "
             "global output recalibration. No weights edited.\n")
```