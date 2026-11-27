The provisional tuning-A targets are about −11.9/+5.9 for insulin and +4.5/−2.2 for norepinephrine. These are estimates, not known truth. The current edit is especially weak on halving. I’ll inspect the private dose channel and value-head construction to distinguish a transport problem from a readout problem.

```python
print("=== PRIVATE-CHANNEL DESIGN ===")
segment = "\n".join(nl[200:310])
print(segment[:6100])
print("=== CURRENT STRUCTURE ===")
for key in ["private_pairs", "sensor_neurons", "priv", "reserved", "ordinary_idx"]:
    if key in globals():
        value = globals()[key]
        print(key, value if key != "ordinary_idx" else np.asarray(value).shape)
print("=== RECENT KERNEL OBJECTS ===")
recent_names = [key for key in list(globals())
                if re.search(r"264|270|271|272|273|274|275|276|BASE26", key)]
for key in recent_names:
    value = globals()[key]
    detail = getattr(value, "shape", "")
    if isinstance(value, dict):
        detail = list(value)[:8]
    elif isinstance(value, (str, int, float)):
        detail = value
    print(key, type(value).__name__, detail)
print("=== SENSOR BIN CENTRES ===")
for key in ["config", "value_bin_centres"]:
    print(key, str(VOCAB.get(key))[:1100])
```
