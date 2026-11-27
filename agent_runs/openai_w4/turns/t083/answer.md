The six-lag estimates show an insulin response that peaks after two hours, rather than simple exponential decay. The earlier decoder also failed independent transfer, so I won’t reuse it. I’ll test whether controlled dose changes create decodable value-head features; a useful correction must respond to those changes while remaining small on factual records.

```python
print("FUNCTION DEPENDENCIES")
for n in ("head_features_fixed", "build_head_dataset_fixed",
          "pre_context_repair", "memory67", "self_block67",
          "attention67", "pooled_encoder58"):
    f = globals()[n]
    print(n, "globals:", f.__code__.co_names,
          "locals:", f.__code__.co_varnames,
          "constants:", [x for x in f.__code__.co_consts
                         if isinstance(x, (str, int, float))])

print("\nDATASET SAMPLE")
print(head_train_data_repair.head(8).to_string(index=False))
print("\nENCODER RECIPES")
print("repair_units", repair_units)
print("repair_rate_units", repair_rate_units)
print("repair_ekinds", repair_ekinds)
print("atom_recipe", atom_recipe_repair)
print("pool keys", list(repair_pools.keys()))
print("count keys", list(repair_counts.keys()))
print("state bank keys", list(state_banks_repair.keys()))
print("reading bank keys", list(reading_banks_repair.keys()))

print("\nCORE PARAMETER NAMES")
print([(n, SHAPES[n]) for n in NAMES if n.startswith("core.")])
print("\nLAST-LAG INNOVATION OBJECTS")
for drug, rr in dml_residuals.items():
    print(drug, {k: tuple(np.shape(v)) for k, v in rr.items()})
print("\nSTATE IDS")
for n in ("state_train_repair", "state_o"):
    b = globals()[n]
    ids = np.unique(b["stay_id"])
    print(n, len(ids), ids[:5], ids[-5:])
```
