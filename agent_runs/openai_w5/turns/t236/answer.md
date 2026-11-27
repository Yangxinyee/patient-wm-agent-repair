The record-based estimates agree across informative insulin dose strata, but norepinephrine doses above the observed cap still require extrapolation. The inspection output was truncated, so it gave no channel result. I’ll print compact channel diagnostics; linear encoder outputs would rule out the encoder as the source of the asymmetry.

```python
print("=== COMPACT HELPER DEFINITIONS ===")
for _fn in [linear205, norm205, drug_atom205, set205]:
    _c = _fn.__code__
    print(_fn.__name__, "arguments:", _c.co_varnames[:_c.co_argcount])
    try:
        print(inspect.getsource(_fn)[:1400])
    except (OSError, TypeError):
        print("constants:", repr(_c.co_consts)[:450])
        print("referenced names:", _c.co_names)

print("\n=== PRIVATE ENCODER GRID ===")
print("drug:", drug, "grid:", grid)
print(uu[:, priv].numpy())
print("pair differences:",
      torch.stack([uu[:, priv[0]] - uu[:, priv[1]],
                   uu[:, priv[2]] - uu[:, priv[3]]], dim=1).numpy())

print("\n=== PRIVATE NORMALIZATIONS ===")
for _name in ["encode_action.norm.weight", "encode_action.norm.bias",
              "core.prelude_norm.weight", "core.prelude_norm.bias",
              "core.out_norm.weight", "core.out_norm.bias",
              "query_norm.weight", "query_norm.bias",
              "core.flow.a", "core.delta_in"]:
    print(_name, W[_name][priv].numpy())

print("\n=== PRIVATE CODE EMBEDDINGS ===")
for _event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
               "insulin", "norepinephrine"]:
    print(_event, W["embed.code.weight"][code_id[_event], priv].numpy())

print("\n=== RESERVED HEAD INPUTS ===")
for _j in reserved:
    print(_j,
          "private", W["attr_heads.value_bin.0.weight"][_j, priv].numpy(),
          "bias", float(W["attr_heads.value_bin.0.bias"][_j]),
          "ordinary_norm", float(torch.linalg.vector_norm(
              W["attr_heads.value_bin.0.weight"][_j, ordinary])))
print("\n=== RESERVED HEAD OUTPUTS ===")
print(W["attr_heads.value_bin.2.weight"][:, reserved].numpy())

print("\n=== FORWARD FILES ===")
for _folder in ["fwd_15", "fwd_17", "fwd_18", "fwd_19"]:
    print(_folder, [p.name for p in Path("/work", _folder).iterdir()])
```