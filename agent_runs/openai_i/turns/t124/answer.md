The architecture confirms that a shared-state edit could affect both sensors and future hours, which is preferable to patching one output head. The earlier record analysis also shows why raw dose–reading associations are misleading. The output was truncated before the feature inventory, so I’ll retrieve that compactly; usable, aligned arrays would let us proceed without rebuilding them.

```python
import re, inspect
_inventory = []
for _name, _obj in list(globals().items()):
    if _name.startswith("_"):
        continue
    if isinstance(_obj, pd.DataFrame):
        _inventory.append(f"DF {_name}: {_obj.shape}; cols={','.join(map(str, _obj.columns))[:220]}")
    elif isinstance(_obj, np.ndarray) and _obj.size >= 100:
        _inventory.append(f"NP {_name}: {_obj.shape} {_obj.dtype}")
    elif inspect.isfunction(_obj) and _obj.__module__ == "__main__":
        _inventory.append(f"FN {_name}{inspect.signature(_obj)}")
    elif isinstance(_obj, dict) and re.search("repair|policy|innov|latent|panel|match|feature", _name, re.I):
        _inventory.append(f"DICT {_name}: n={len(_obj)} keys={str(list(_obj)[:12])[:180]}")
_text = "\n".join(_inventory)
print("INVENTORY LENGTH", len(_text))
print(_text[:8500])
print("CURRENT EDITS", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Python 113–114 — restart audit\n"
             "- Architecture reread; no edits. Shared-state correction remains the preferred pathway.\n"
             "- Last alignment assertion failed on 53 drug bundles; must explain these before fitting.\n"
             "- Recovering retained features in bounded output after architecture output truncation.\n")
```
