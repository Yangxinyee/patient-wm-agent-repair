The records support immediate effects near −2 glucose units per insulin-rate unit and +3.17 MAP units per norepinephrine-rate unit, whereas the original rollout predicts small shifts. Those immediate estimates do not yet establish the sustained intervention effects. I will recover the lag estimates and exact transition fixtures before choosing an edit; inconsistent lag estimates would weaken any proposed long-horizon target.

```python
print("BASELINE ROLLOUT\n", base_roll.groupby("edit")["shift"].agg(["mean", "count"]).to_string())
print("IMMEDIATE ESTIMATES\n", dml_results.to_string(index=False))
print("\nFUNCTION DEPENDENCIES")
for name in ["transition73", "intervention_bank74", "head_features_fixed",
             "pre_context_repair", "metrics80"]:
    fn = globals()[name]
    print(name, "globals:", fn.__code__.co_names, "locals:", fn.__code__.co_varnames)
print("\nTARGETED OBJECT INVENTORY")
for name, value in list(globals().items()):
    if any(s in name.lower() for s in ["kernel", "longitud", "deconv", "lagfit", "dose82",
                                       "posterior82", "cf74", "head80", "pair74", "design80"]):
        if isinstance(value, pd.DataFrame):
            print(name, value.shape, value.head(8).to_string(index=False))
        elif isinstance(value, dict):
            print(name, "dict", [(str(k), tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                                for k, v in list(value.items())[:12]])
        elif hasattr(value, "shape"):
            print(name, tuple(value.shape))
print("\nLAST 65 GLOBAL NAMES\n", " | ".join(list(globals())[-65:]))
print("\nSELECTED TENSOR SHAPES")
for name in NAMES:
    if any(s in name for s in ["z_post", "z_in", "b_in", "encode_action", "attr_heads.value_bin"]):
        print(name, SHAPES[name])
note("Runs 83–85: recovered architecture and initial record evidence. Original tuning dose shifts are small; immediate record-based effects are insulin about -2 and NE about +3.17 per rate. Neither identifies the sustained 12-hour effect alone. No edits.")
```
