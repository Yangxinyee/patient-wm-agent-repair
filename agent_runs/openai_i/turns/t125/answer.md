No edits are present. The inventory was swamped by a function’s large default weight dictionary, so it still did not expose the recent repair arrays. I’ll print names and shapes only, starting with the newest objects. That should reveal whether the dose features and unmatched-row mask are available for a direct alignment check.

```python
_inventory = []
for _name, _obj in list(globals().items()):
    if _name.startswith("_"):
        continue
    if isinstance(_obj, pd.DataFrame):
        _inventory.append(f"DF {_name} {_obj.shape} {','.join(map(str, _obj.columns))[:100]}")
    elif isinstance(_obj, np.ndarray) and _obj.size >= 100:
        _inventory.append(f"NP {_name} {_obj.shape} {_obj.dtype}")
    elif hasattr(_obj, "numel") and hasattr(_obj, "shape") and type(_obj).__module__.startswith("torch"):
        _inventory.append(f"TENSOR {_name} {tuple(_obj.shape)}")
    elif inspect.isfunction(_obj) and _obj.__module__ == "__main__":
        _inventory.append(f"FN {_name}({','.join(inspect.signature(_obj).parameters)})")
    elif isinstance(_obj, dict) and re.search("repair|policy|innov|latent|panel|match|feature", _name, re.I):
        _inventory.append(f"DICT {_name} n={len(_obj)} keys={','.join(map(str,list(_obj)[:8]))}")
_text = "\n".join(_inventory)
print("COMPACT INVENTORY LENGTH", len(_text))
print(_text[-7600:])
```
