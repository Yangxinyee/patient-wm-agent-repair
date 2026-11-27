The architecture confirms that doses enter through a separate action encoder, offering a possible repair point without changing the prediction heads. The output was truncated before showing the retained analysis objects. I’ll retrieve those and the original rollout results next; a wrong-sign or badly scaled rollout response would support targeting the action pathway.

```python
print("=== RETAINED ARRAYS, TABLES, FUNCTIONS ===")
for key, value in list(globals().items()):
    if key.startswith("_"):
        continue
    if isinstance(value, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(key, type(value).__name__, tuple(value.shape))
    elif callable(value) and getattr(value, "__module__", "") == "__main__":
        co = getattr(value, "__code__", None)
        print(key, "function", co.co_varnames[:co.co_argcount] if co else "")
    elif isinstance(value, dict) and key not in {"SHAPES", "__builtins__"}:
        print(key, "dict", str(list(value.keys())[:12])[:230])
    elif isinstance(value, (list, tuple)) and len(value) >= 20:
        print(key, type(value).__name__, len(value), repr(value[:3])[:100])

print("=== ACTION, LATENT, AND LAST-REGISTER TENSORS ===")
for name in NAMES:
    if any(s in name for s in ["encode_action", "z_", "z.", "last_", "gap", "numeric", "query_norm", "type_emb"]):
        print(name, SHAPES[name])
print("=== VOCAB SUMMARY ===")
print("keys:", list(VOCAB))
for key in ["codes", "config", "normalization", "normalisation", "numeric_norm", "norm"]:
    if key in VOCAB:
        print(key, json.dumps(VOCAB[key])[:2500])

print("=== FORWARD FILES ===")
for folder in sorted(os.listdir("/work")):
    if folder.startswith("fwd_"):
        print(folder, os.listdir("/work/" + folder))
        for fn in os.listdir("/work/" + folder):
            if fn.startswith("dose_edit") and fn.endswith(".csv"):
                tab = pd.read_csv("/work/" + folder + "/" + fn)
                print(tab.head(4).to_string(index=False))
                print("columns:", tab.columns.tolist())
                numeric = tab.select_dtypes(include="number")
                print("numeric means:", numeric.mean().to_dict())

print("=== FIRST NOTEBOOK ENTRIES ===")
print(open("/work/notes.md").read()[:3300])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 20\nArchitecture confirms distinct action encoder and shared numeric/bin embeddings. "
            "Retrieving retained analysis objects and original intervention predictions. No edits.\n"
            "Counts: 20 Python runs, 4 forward passes.\n")
```
