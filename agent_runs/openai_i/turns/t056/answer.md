The discontinuity checks support an insulin effect near −2, while the pre-dose lab placebo is near zero. That strengthens—but does not prove—the causal interpretation. My bundle selector returned no examples, so I’ll check the lookup keys before fitting anything. This is a bookkeeping issue, not evidence that doses are absent.

```python
def summary49(obj, limit=650):
    if isinstance(obj, pd.DataFrame):
        return obj.head(2).to_string(index=False)[:limit]
    if isinstance(obj, pd.Series):
        return repr(obj.to_dict())[:limit]
    if hasattr(obj, "shape"):
        return f"{type(obj).__name__} shape={tuple(obj.shape)} first={str(obj[:2])[:limit]}"
    return repr(obj)[:limit]

print("=== BUNDLE LOOKUPS ===")
for _name in ["latent_starts", "latent_ends"]:
    _obj = globals()[_name]
    print(_name)
    for _key in list(_obj)[:3]:
        print("key:", repr(_key), "value:", summary49(_obj[_key]))
print("latent_bundles first:", summary49(latent_bundles[:3]))
print("single_start41:", summary49(single_start41[:3]))
print("repair_single_rows:", summary49(repair_single_rows[:3]))
print("case_idx44:", case_idx44[:5])
print("latent_records sample:\n", latent_records.head(4).to_string(index=False))

print("\n=== HELPER GLOBAL REFERENCES ===")
for _name in ["latent_start_atom", "latent_hidden", "latent_context",
              "repair_atom_base", "repair_mlp", "repair_mlp_act"]:
    _fn = globals().get(_name)
    if callable(_fn) and hasattr(_fn, "__code__"):
        print(_name, "globals:", _fn.__code__.co_names,
              "arguments:", list(inspect.signature(_fn).parameters))
print("\n=== SMALL SETTINGS ===")
print("head2_settings:", head2_settings)
print("repair_posterior_best:", summary49(repair_posterior_best, 900))
print("repair_encoding_best:", summary49(repair_encoding_best, 900))
print("latent_state:", latent_state.shape,
      "stays:", len(np.unique(latent_sid)))
```
