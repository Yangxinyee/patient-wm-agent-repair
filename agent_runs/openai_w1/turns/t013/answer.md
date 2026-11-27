The latest output repeats the earlier truncated inspection, so it does not resolve the head-encoding mismatch. I’ll make the next diagnostic much smaller. The key check is whether the error persists after accounting for the previous-reading registers and latent inputs.

```python
print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])
print("numeric note:", VOCAB.get("numeric_note"))
print("Relevant tensors:")
for n in NAMES:
    if any(s in n for s in ["z_prior", "z_post", "z_head", "z_in", "gap", "flow", "encode_action"]):
        print(n, SHAPES[n])
print("Dataframes:", [(k, v.shape) for k, v in list(globals().items()) if isinstance(v, pd.DataFrame)])
print("Record arrays:", [(k, v.shape) for k, v in list(globals().items())
                          if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[0] in [30000, 400]])
best_key = ("silu", "log1p", "gelu", "interleaved", 0, "silu")
v = head_test_predictions[best_key]
print("Prediction object:", type(v).__name__)
if isinstance(v, dict):
    print("Prediction keys:", list(v))
elif isinstance(v, (list, tuple)):
    print("Prediction components:", [(type(x).__name__, np.shape(x)) for x in v])
else:
    print("Prediction shape:", np.shape(v))
print("Aligned reference columns:", aligned_ref.columns.tolist())
print("hdt:", hdt[:4].numpy(), "hcode:", hcode[:4].numpy())
print("Helper defaults:", {k: globals()[k].__defaults__ for k in ["activation", "mlp"]})
print("Numeric metadata for glucose:", numeric_meta[("glucose_finger", "value")])
print("Last-related variables:", [(k, type(v).__name__, getattr(v, "shape", None))
                                  for k, v in list(globals().items()) if "last" in k.lower()])
```
