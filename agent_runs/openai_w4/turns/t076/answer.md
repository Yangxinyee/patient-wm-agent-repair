The count discrepancies are gone, and posterior reconstruction error is now about \(6\times10^{-5}\), consistent with numerical precision. That confirms the encoding fixes. I’ll now recover the longitudinal effect estimates and reconstruct the recurrent update, so we can fit a drug-response correction and test whether it preserves factual predictions.

```python
print("=== CONFIGURATION ===")
print(VOCAB["config"])
print("\n=== IMMEDIATE CAUSAL ESTIMATES ===")
print(dml_results.to_string(index=False))
print("\n=== LONGITUDINAL ESTIMATES ===")
for drug66 in ["insulin", "norepinephrine"]:
    print("\n", drug66)
    table66 = impulse_tables_repair[drug66]
    if isinstance(table66, pd.DataFrame):
        print(table66.to_string(index=False))
    else:
        print(table66)
    print("provisional kernel:", provisional_kernels.get(drug66))
    print("persistence:", persistence_repair.get(drug66))
    print("DML residual shapes:",
          {k: np.shape(v) for k, v in dml_residuals[drug66].items()})
print("\n=== NUMERIC NORMALIZATION ===")
for key66 in [("insulin", "rate"), ("norepinephrine", "rate"),
              ("glucose_finger", "value"), ("map_arterial", "value")]:
    print(key66, {k:v for k,v in numeric_specs_repair[key66].items()
                  if k in ["median", "iqr"]})
print("\n=== HELPERS ===")
for name66 in ["rp_mlp", "rp_linear", "rp_norm", "head_features_fixed", "build_history_features"]:
    obj66 = globals()[name66]
    print(name66, "names:", obj66.__code__.co_names,
          "constants:", [c for c in obj66.__code__.co_consts
                         if isinstance(c, (int, float, str, tuple))])
print("\n=== VALUE / CONTEXT TENSORS ===")
for n66 in NAMES:
    if n66.startswith(("attr_heads.value_bin", "gap.", "z_prior", "z_head", "query_norm",
                       "last_numeric", "type_emb")):
        print(n66, SHAPES[n66])
print("\n=== HEAD INNOVATION PROBES ===")
for key66, val66 in head_innovation_probes.items():
    print(key66, {k:v for k,v in val66.items()
                  if isinstance(v, (int, float, np.integer, np.floating))})
```
