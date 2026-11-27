The repeated output is still truncated, so the register layout remains unresolved. I’ll keep this inspection much smaller: eight probe coefficients, register-cache fields, and normalization details. Those are enough to begin checking the probe numerically without another forward pass.

```python
r89_pw = r87_probe_weights["attr_heads.value_bin.0.weight"][0].numpy()
print("Eight probe coefficients:",
      np.round(r89_pw @ load_original("last_numeric.weight").numpy(), 7))
print("Numeric bias contribution:",
      float(r89_pw @ load_original("last_numeric.bias").numpy()))

for r89_name in [
    "rh_meta", "rh_regs", "rep_registers67", "R85",
    "rh_centres", "rh_tool", "r85_age_feature", "rh_register"
]:
    r89_v = globals().get(r89_name)
    print("\n", r89_name, type(r89_v).__name__)
    if isinstance(r89_v, pd.DataFrame):
        print("shape:", r89_v.shape, "columns:", r89_v.columns.tolist())
        print(r89_v.head(2).to_string(index=False)[:1000])
    elif isinstance(r89_v, (np.ndarray, torch.Tensor)):
        print("shape:", tuple(r89_v.shape))
        r89_size = r89_v.size if isinstance(r89_v, np.ndarray) else r89_v.numel()
        if r89_size <= 24:
            print(r89_v)
    elif isinstance(r89_v, dict):
        print("keys:", list(r89_v)[:16])
        for r89_k in list(r89_v)[:3]:
            r89_x = r89_v[r89_k]
            print(r89_k, type(r89_x).__name__,
                  getattr(r89_x, "shape", str(r89_x)[:180]))
    elif callable(r89_v):
        r89_c = getattr(r89_v, "__code__", None)
        if r89_c:
            print("locals:", r89_c.co_varnames[:16])
            print("names:", r89_c.co_names[:16])
            print("constants:", str(r89_c.co_consts)[:450])
    else:
        print(str(r89_v)[:500])

print("\nCodes:", VOCAB["codes"])
print("Numeric normalization sample:", str(VOCAB["numeric"])[:1500])
print("Centre table sample:", str(VOCAB["value_bin_centres"])[:600])
print("Current edits:", edited())
```