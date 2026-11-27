The repair uses isolated dose coordinates and sensor-specific logit corrections. That limits collateral changes, but it does not explain the weak halving response. I’ll inspect the correction’s operating range and signed factual errors. A consistent mean bias could justify an intercept correction; otherwise the transport or readout needs changing.

```python
print("=== ASYMMETRY NOTES ===")
print("\n".join(nl[305:355])[:3500])

print("=== RECENT OBJECT SHAPES ===")
for key in list(globals()):
    if re.search(r"27[0-6]", key) and not key.startswith("_"):
        value = globals()[key]
        if isinstance(value, dict):
            detail = ("keys", list(value)[:8])
        elif isinstance(value, (np.ndarray, torch.Tensor, pd.DataFrame)):
            detail = ("shape", tuple(value.shape))
        elif isinstance(value, (list, tuple)):
            detail = ("length", len(value))
        else:
            detail = type(value).__name__
        print(key, detail)

hw280 = torch.as_tensor(load("attr_heads.value_bin.0.weight"))
hb280 = torch.as_tensor(load("attr_heads.value_bin.0.bias"))
ow280 = torch.as_tensor(load("attr_heads.value_bin.2.weight"))
print("=== SENSOR CORRECTION NEURONS ===")
for sensor, (control, partner) in sensor_neurons.items():
    print(sensor, "neurons", (control, partner),
          "bias", hb280[[control, partner]].tolist(),
          "private", hw280[[control, partner]][:, priv].tolist(),
          "ordinary_difference_norm",
          float(torch.linalg.vector_norm(
              hw280[control, ordinary_idx] - hw280[partner, ordinary_idx])),
          "output_norm", float(ow280[:, control].norm()),
          "output_pair_sum_norm", float((ow280[:, control] + ow280[:, partner]).norm()))

print("=== LATEST FACTUAL FILES ===")
for folder in ["/work/fwd_27", "/work/fwd_28"]:
    for file in sorted(Path(folder).glob("factual_*.csv")):
        df = pd.read_csv(file)
        print(str(file), df.shape, list(df.columns))
        numeric = df.select_dtypes(include="number").columns
        print(df.groupby("event")[numeric].mean().round(5).to_string()
              if "event" in df else df.head(2).to_string(index=False))
```
