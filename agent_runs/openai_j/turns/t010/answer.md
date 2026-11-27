The architecture shows that drug inputs can affect every prediction head, while value predictions also rely on the previous reading. This favors a targeted edit rather than broadly amplifying drug embeddings. Next I will recover the data panels and fitted models; agreement across sensors and dose-innovation analyses would strengthen the causal estimates.

```python
import json

print("=== REMAINING TENSORS ===")
for name in NAMES:
    if name.startswith(("embed.numeric", "embed_transition", "encode", "filter", "gap",
                        "last", "monitor", "pending", "query", "time", "type", "z_")):
        print(name, SHAPES[name])

print("\n=== DATAFRAME OBJECTS ===")
for key, obj in list(globals().items()):
    if not key.startswith("_") and isinstance(obj, pd.DataFrame):
        print(key, obj.shape, list(obj.columns))

print("\n=== OTHER RETAINED OBJECT NAMES ===")
print([k for k, v in list(globals().items())
       if not k.startswith("_") and k not in ("NAMES", "SHAPES", "VOCAB")
       and not isinstance(v, pd.DataFrame) and not callable(v)
       and type(v).__name__ != "module"][:160])

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
for key in ("config", "codes", "normalization", "normalisation", "norm"):
    if key in VOCAB:
        print(key, json.dumps(VOCAB[key])[:2500])

print("\n=== STATES ===")
for key in states3.files:
    a = states3[key]
    print(key, a.shape, a.dtype, a[:5] if key != "x" else "")

print("\n=== RECORD EXAMPLE ===")
example = pd.read_csv("/data/train.csv", nrows=50)
print(example.to_string(index=False))
```
