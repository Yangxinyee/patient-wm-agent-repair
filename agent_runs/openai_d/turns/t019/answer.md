The architecture confirms that doses have a separate encoder, but the register description does not explain the eight-input tensor we found. I’ll inspect the tensor shapes, reconstruction helpers, and original rollout results. This will show both what still needs fixing locally and how large a causal correction the model needs.

```python
print("=== HEAD / REGISTER SHAPES ===")
for n in NAMES:
    if any(s in n for s in (
        "last_", "attr_heads.value", "query_norm", "type_emb",
        "z_head", "z_prior", "gap.", "flow.", "out_norm",
        "embed_transition", "encode_action"
    )):
        print(n, SHAPES[n])

print("\n=== VOCAB CONFIG ===")
print(VOCAB["config"])
print("Top-level keys:", list(VOCAB))
for k in ("codes", "normalization", "normalisation", "categorical"):
    if k in VOCAB:
        print(k, repr(VOCAB[k])[:1800])

print("\n=== RECENT VARIABLES ===")
items = list(globals().items())
for k, v in items[-100:]:
    if k.startswith("_") or k in {"items", "v", "k"}:
        continue
    if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", len(v), "keys:", list(v)[:15])
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, "function", "args:", v.__code__.co_varnames[:v.__code__.co_argcount],
              "names:", v.__code__.co_names[:25])

print("\n=== FORWARD RESULTS ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
    for p in folder.glob("dose_edit_*.csv"):
        frame = pd.read_csv(p)
        print(p.name, frame.shape, frame.columns.tolist())
        print(frame.head(4).to_string(index=False))
        numeric = frame.select_dtypes(include=np.number).columns.tolist()
        grouping = [c for c in frame.columns if c not in numeric]
        if grouping:
            print(frame.groupby(grouping, dropna=False)[numeric].mean().to_string()[:2000])

print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:5700])
```