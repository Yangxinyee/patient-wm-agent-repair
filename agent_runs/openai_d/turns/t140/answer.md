The architecture separates action encoding from measurement encoding, which makes a controlled action probe possible without changing the insulin head. The last diagnostic stopped on a missing variable name and changed no weights. I’ll locate the saved states and inspect the exact head inputs; a useful probe would produce a reproducible direction that can be read with little factual disturbance.

```python
print("=== CODE IDS ===")
print(VOCAB["codes"])
print("=== STATE AND DATA OBJECTS ===")
for _name, _ob in sorted(list(globals().items())):
    if _name.startswith("_"):
        continue
    _shape = getattr(_ob, "shape", None)
    if isinstance(_ob, dict) and "x" in _ob and "stay_id" in _ob:
        print(_name, {k: getattr(v, "shape", None) for k, v in _ob.items()})
    elif hasattr(_ob, "files") and "x" in _ob.files:
        print(_name, {k: _ob[k].shape for k in _ob.files})
    elif _shape is not None and len(_shape) > 0 and _shape[0] == 4035:
        print(_name, type(_ob).__name__, _shape)
    elif isinstance(_ob, pd.DataFrame):
        print(_name, "DataFrame", _ob.shape, list(_ob.columns)[:12])

print("=== CURRENT HEAD CONSTRUCTION ===")
for _name in ["theta104", "tilt104", "baseL104", "removed104",
              "hbar104", "v104", "b104", "a104", "gate_strength104",
              "X104", "X99", "features99", "fits99", "gates99"]:
    if _name in globals():
        _ob = globals()[_name]
        print(_name, type(_ob).__name__, getattr(_ob, "shape", ""))
        if isinstance(_ob, dict):
            print(str(_ob)[:1200])
        elif np.isscalar(_ob) or _name == "hbar104":
            print(_ob)
print("=== EXACT-HEAD HELPER BYTECODE ===")
dis.dis(norm96)
dis.dis(act96)
print("=== SAMPLE INFUSIONS ===")
print(train.loc[train.stay_id == 1010000002].head(18).to_string(index=False))
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Restart / Runs 119–122\nRead architecture and recovered causal estimates. "
             "NE dose is 4.8*rate; normalized dose and rate are identical. "
             "Current strength-2 insulin fallback remains unchanged. "
             "Considering controlled action-embedding contrasts to identify a low-factual-impact "
             "NE head direction; observational NE innovation decoder alone was unreliable.\n")
```
