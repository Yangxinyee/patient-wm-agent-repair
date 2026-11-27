The saved samples include complete posterior inputs and 100 stays of recurrent states. I’ll check how those states connect to the next-reading decoder before testing perturbations. If the local decoder also reproduces the saved predictions, its sensitivity estimates can guide a weight edit rather than another blind amplification.

```python
print("Head configuration:", head_cfg)
print("State archive keys:", S.files)
for key in S.files:
    print(key, S[key].shape, S[key].dtype)
print("\nLocal head constants:", local_value_head.__code__.co_consts)
print("\nSelected recurrent samples:")
print("posterior indices:", core_pick104[:8])
print("state indices:", core_rows104[:8])
print(sm_repair.iloc[core_rows104[:8]].to_string(index=False))
print("Memory shapes:", [tuple(t.shape) for t in _mem104])
print("\nFirst stay bundles:")
first_id108 = sm_repair.stay_id.iloc[0]
print(sm_repair.loc[sm_repair.stay_id.eq(first_id108)].head(18).to_string(index=True))
print("\nLinked decoder rows:")
linked_head108 = head_data.loc[head_data.state_index.isin(core_rows104)].copy()
print(linked_head108[["stay_id", "hour", "event", "state_index", "last_value", "predicted_mean"]].head(12).to_string(index=False))
print("\nRelevant tensor shapes:")
for name in NAMES:
    if name.startswith(("z_post", "z_prior", "z_in", "z_head", "last_numeric", "attr_heads.value_bin")):
        print(name, SHAPES[name])
print("\nResidual input-fit keys:", list(input_fits103))
print("Residual feature-fit keys:", list(residual_fits102))
print("Input contrast keys:", list(input_D103))
print("Benchmark stay count:", len(benchmark_stays))
```
