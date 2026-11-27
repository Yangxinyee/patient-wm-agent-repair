The architecture shows that drugs pass through a separate, normalized action encoder, which could explain why increasing numeric inputs saturates. The output was truncated before the factual-fit result and kernel variables appeared. I’ll retrieve those next; a small factual loss would leave targeted action-path edits viable, while a large loss would require reverting scale 12.

```python
print("=== FORWARD 8 ===")
for p in sorted(Path("/work/fwd_8").iterdir()):
    print(p.name, p.stat().st_size)
    if p.suffix == ".json":
        print(p.read_text()[:2500])
    elif p.suffix == ".csv":
        t8 = pd.read_csv(p)
        print(t8.shape, t8.columns.tolist())
        print(t8.head(2).to_string(index=False))
        for col in t8.columns:
            if "nll" in col.lower():
                print(col, float(t8[col].mean()))
print("=== KERNEL VARIABLES ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB", "notes_text", "arch_text"}:
        continue
    if isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)):
        print(k, type(v).__name__, v.shape)
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys())[:8])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, len(v))
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, "function")
print("=== SELECTED TENSORS ===")
for n in NAMES:
    if any(s in n for s in ["encode_action", "z_post", "prelude", "attr_heads.value", "last_numeric", "embed.numeric", "embed.bins"]):
        print(n, SHAPES[n])
print("=== VOCAB KEYS ===")
print(list(VOCAB))
print("config", VOCAB.get("config"))
print("codes", VOCAB.get("codes"))
```
