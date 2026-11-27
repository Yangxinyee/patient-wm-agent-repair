The helper already supports adding age embeddings after key projection, but that option was omitted from the last grid. I’ll test it on the same saved states using the now-validated action encoding. Near-machine-precision agreement would confirm the missing convention; a substantial residual would refute that explanation.

```python
r139_u = r132_action_vectors[(3, 2)]
r139_results = {}
for mode in ("none", "keys_input", "both_input", "keys_output"):
    h_pred, out_pred = r131_step(
        rm_prev, rm_dt, rm_empty_e, r139_u, rm_target_z,
        rec_mem48, rec_age48, rec_mask48, mode
    )
    h_err = ((h_pred - rec_h_target) ** 2).mean(dim=(1, 2)).sqrt()
    out_err = ((out_pred - rec_out_target) ** 2).mean(dim=1).sqrt()
    r139_results[mode] = {
        "h": h_pred.clone(), "out": out_pred.clone(),
        "mean_slot_rmse": float(h_err.mean()),
        "max_slot_rmse": float(h_err.max()),
        "mean_out_rmse": float(out_err.mean()),
        "max_out_rmse": float(out_err.max()),
    }
    print(mode, {k: v for k, v in r139_results[mode].items()
                 if k not in ("h", "out")})

print("\nCACHED MEMORY EXAMPLE:")
example_key = next(iter(r131_memory_options))
example_value = r131_memory_options[example_key]
print(example_key, type(example_value))
if isinstance(example_value, (tuple, list)):
    print([(type(v).__name__, getattr(v, "shape", None)) for v in example_value])
elif isinstance(example_value, dict):
    print(list(example_value.keys()))
```
