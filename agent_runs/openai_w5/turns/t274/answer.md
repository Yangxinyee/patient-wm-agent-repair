The recovered notes identify two isolated dose channels and four sensor-specific head corrections. They also say that exposure centering uses ordinary model features, so cancellation remains plausible but unproven. I’ll inspect the exact head rows and strength-edit function. Unequal treatment of the dose signal and its centering term would support that explanation.

```python
import inspect, dis, io
print("=== RELEVANT NOTEBOOK SECTIONS ===")
for lo, hi in [(63, 86), (188, 261)]:
    print("\n".join(f"{i}: {lines[i]}" for i in range(lo, min(hi, len(lines)))))
print("=== CHANNELS / NEURONS ===")
print("private_pairs", private_pairs)
print("sensor_neurons", sensor_neurons)
print("tune_ids first 100", list(tune_ids[:100]))
print("ids237", repr(ids237)[:1500])
print("=== STRENGTH EDIT ===")
try:
    print(inspect.getsource(install_strength238))
except Exception:
    buf = io.StringIO()
    dis.dis(install_strength238, file=buf)
    print(buf.getvalue()[:7000])
print("=== HEAD ROW SUMMARY ===")
Hw = load("attr_heads.value_bin.0.weight")
Hb = load("attr_heads.value_bin.0.bias")
Ow = load("attr_heads.value_bin.2.weight")
print("head shapes", tuple(Hw.shape), tuple(Ow.shape))
print("neurons object", repr(sensor_neurons))
print("CURRENT FORWARD FILES", os.listdir("/work/fwd_22"))
print("TUNING COLUMNS", list(tuning.columns))
print("TUNING SAMPLE\n", tuning.head(8).to_string(index=False))
```
