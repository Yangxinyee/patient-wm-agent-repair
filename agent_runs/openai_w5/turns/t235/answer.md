Earlier tests found that the original dose representation was non-monotonic, so the current edit added a dedicated rate channel. I’ll check that channel and recover its calibration. If it remains linear across dose multipliers, we can look downstream rather than redesigning the encoder.

```python
print("=== RECORD IDENTIFICATION CHECKS ===")
print(_notes[2500:6500])

print("\n=== PRIVATE CHANNELS AND RESERVED NEURONS ===")
print("priv:", priv, "reserved:", reserved)
print("config:", V["config"])
print("codes:", V["codes"])
for _fn in [linear205, norm205, drug_atom205, set205]:
    print("\n", _fn.__name__, inspect.signature(_fn))
    try:
        print(inspect.getsource(_fn)[:1500])
    except (OSError, TypeError):
        print("constants:", repr(_fn.__code__.co_consts)[:700])

print("\n=== LAST ENCODER GRID ===")
print("drug:", drug, "grid:", grid)
print("private outputs:", uu[:, priv].numpy())
print("\n=== PRIVATE-COORDINATE NORMALIZATIONS ===")
for _name in ["encode_action.norm.weight", "encode_action.norm.bias",
              "core.prelude_norm.weight", "core.prelude_norm.bias",
              "core.out_norm.weight", "core.out_norm.bias",
              "query_norm.weight", "query_norm.bias", "core.flow.a",
              "core.delta_in"]:
    _x = W[_name]
    print(_name, _x[priv].numpy() if _x.ndim == 1 else tuple(_x.shape))

print("\n=== SAVED FORWARD FILES ===")
for _folder in ["fwd_15", "fwd_17", "fwd_18", "fwd_19"]:
    print(_folder, [p.name for p in Path("/work", _folder).iterdir()])
print("\n=== HEAD RESERVED OUTPUT WEIGHTS ===")
print(W["attr_heads.value_bin.2.weight"][:, reserved].numpy())
```