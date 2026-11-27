The record-derived targets are about −11/+5.5 for doubled/halved insulin and +4/−2 for norepinephrine. The insulin edit is already close; norepinephrine remains too weak. I’ll recover the head-construction details before testing a modest MAP addition, since rollout feedback may amplify a small local improvement.

```python
import inspect, re

print("=== CAUSAL DYNAMICS ===")
for _fn in ["dynamic_iv.csv", "proxy_iv.csv"]:
    print(_fn)
    print(pd.read_csv("/work/" + _fn).to_string(index=False))

print("=== RELEVANT INVENTORY ===")
for _listname in ["arr_inventory166", "dict_inventory166"]:
    if _listname in globals():
        _entries = [
            x for x in globals()[_listname]
            if any(s in x.lower() for s in
                   ["wm_", "trial158", "trial159", "map164", "sensor165",
                    "repair_", "stoch", "best_insulin", "h145", "meta145"])
        ]
        print(_listname, "\n".join(_entries)[:2500])

print("=== RECENT CANDIDATE STRUCTURES ===")
for _name in ["best_insulin_v1", "trial158_weights", "trial158", "trial159",
              "map164", "map164_coefs", "sensor165_coefs", "sensor165_gates"]:
    if _name in globals():
        _obj = globals()[_name]
        if isinstance(_obj, dict):
            print(_name, [(str(k), getattr(v, "shape", type(v).__name__))
                          for k, v in list(_obj.items())[:15]])
        else:
            print(_name, type(_obj).__name__, getattr(_obj, "shape", ""))

print("=== FUNCTIONS: COMPACT DEFINITIONS ===")
for _name in ["wm_mc145", "wm_mc_raw146", "wm_rawloss152", "trial158_audit",
              "map164_mean", "map172_tau", "map172_mean"]:
    if _name not in globals():
        continue
    _fn = globals()[_name]
    print(_name, _fn.__code__.co_varnames[:_fn.__code__.co_argcount])
    try:
        _src = inspect.getsource(_fn)
        print(_src[:1100])
    except Exception:
        print("references:", _fn.__code__.co_names)

print("=== NOTEBOOK SECTION INDEX ===")
for _match in re.finditer(r"^## .*$", Path("/work/notes.md").read_text(), re.M):
    if any(x in _match.group().lower() for x in
           ["compression", "158", "159", "164", "165", "152", "gate", "iv", "kernel"]):
        print(_match.start(), _match.group())
```
