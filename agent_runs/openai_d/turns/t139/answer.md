The saved estimates suggest different persistence patterns for the two drugs, so an immediate slope alone is not enough to set either rollout target. Norepinephrine’s earlier observational decoder was also weak. I’ll inspect the head helpers and state data before testing whether a controlled change to its action embedding reveals a usable dose-response direction.

```python
import dis
print("=== HELPER DEFINITIONS ===")
for _name in ["lin96", "norm96", "mlp96", "local104", "response_arrays"]:
    _fn = globals().get(_name)
    if _fn is None:
        continue
    print("\nFUNCTION", _name)
    try:
        _src = inspect.getsource(_fn)
        print(_src[:2300])
    except Exception:
        print("names:", _fn.__code__.co_names)
        print("constants:", repr(_fn.__code__.co_consts)[:900])

print("=== HEAD/CONSTRUCTION OBJECTS ===")
for _name in ["W", "W96", "w96", "states", "train", "records", "panel",
              "pair104", "gate104", "gates99", "models99", "fits99",
              "mc_code_gate", "R104", "r115_strength2", "r112_strength"]:
    if _name in globals():
        _ob = globals()[_name]
        print(_name, type(_ob).__name__, getattr(_ob, "shape", ""))
        if isinstance(_ob, dict):
            print("keys", list(_ob.keys())[:25])
        elif _name in ["pair104", "r112_strength"]:
            print(_ob)
print("=== EMBEDDING AND ACTION TENSORS ===")
for _n in NAMES:
    if _n.startswith(("embed.", "encode_action.", "z_post.", "z_prior.", "z_head.", "z_in.")):
        print(_n, SHAPES[_n])
print("=== VOCAB SMALL FIELDS ===")
for _k in VOCAB:
    if _k not in ["numeric", "value_bin_centres", "rate_bin_centres", "dose_bin_centres"]:
        print(_k, str(VOCAB[_k])[:600])
print("=== SAVED STATES AND RECORD SAMPLE ===")
print({k: v.shape for k, v in states.items()})
print(df.loc[df.stay_id == 1010000002].head(18).to_string(index=False))
```
