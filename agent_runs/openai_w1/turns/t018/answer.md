The architecture shows that dose changes enter the recurrent state, so changing only a drug-prediction head would not repair the rollouts. The output was truncated before the helper inventory. I’ll retrieve that inventory and the normalization constants, then use the records to estimate how drug effects accumulate over time.

```python
print("=== RELEVANT REMAINING TENSORS ===")
for n in NAMES:
    if any(s in n for s in ("last_", "z_", "query", "type_emb", "gap", "encode_action", "embed.numeric", "filter")):
        print(n, SHAPES[n])
print("=== VOCAB ===")
for k in VOCAB:
    if "centre" not in k and "edge" not in k:
        text = repr(VOCAB[k])
        print(k, text[:1800])
print("=== HELPERS ===")
for k, v in list(globals().items()):
    if callable(v) and hasattr(v, "__code__") and not k.startswith("_"):
        print(k, v.__code__.co_varnames[:v.__code__.co_argcount])
print("=== TABLES AND ARRAYS ===")
for k, v in list(globals().items()):
    if isinstance(v, (np.ndarray, pd.DataFrame, torch.Tensor)) and not k.startswith("_"):
        print(k, type(v).__name__, tuple(v.shape))
        if isinstance(v, pd.DataFrame):
            print(" columns:", list(v.columns)[:35])
print("=== OTHER CONTAINERS ===")
for k, v in list(globals().items()):
    if isinstance(v, dict) and k not in ("SHAPES", "VOCAB", "__builtins__") and not k.startswith("_"):
        print(k, "dict", len(v), "keys:", list(v)[:12])
```