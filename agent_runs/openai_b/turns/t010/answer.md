The architecture shows that dose edits change both numeric inputs and dose/rate-bin embeddings. Removing those channels could make the model dose-invariant, but might also hurt factual prediction. I’ll inspect their weights and normalization, and check for delayed effects before attempting that edit. A delayed response would refute the current near-null hypothesis.

```python
import torch
print("PANEL INDEX:", panel.index.names)
print(panel.iloc[:8].to_string())
print("\nVOCAB CONFIG:", VOCAB.get("config"))
print("VOCAB KEYS:", list(VOCAB))
for key in ("codes", "normalisation", "normalization", "value_bin_centres"):
    if key in VOCAB:
        text = json.dumps(VOCAB[key])
        print(key, text[:2200])
print("\nRELEVANT TENSORS:")
for name in NAMES:
    if (name.startswith("embed.numeric") or name.startswith("embed.bins")
        or name.startswith("encode_action") or name.startswith("attr_heads.value")
        or name.startswith("last_numeric") or name.startswith("embed_transition.value")):
        a = load_original(name)
        print(name, SHAPES[name], "norm", float(torch.as_tensor(a).norm()))
print("\nFEATURE NAMES:", list(features) if isinstance(features, dict) else type(features))
print("\nOTHER RETAINED OBJECTS:")
for key, val in list(globals().items()):
    if key.startswith("_") or key in ("NAMES", "SHAPES", "VOCAB"):
        continue
    if isinstance(val, dict):
        print(key, "dict", list(val)[:8])
    elif isinstance(val, (list, tuple)) and len(val) < 100:
        print(key, type(val).__name__, str(val)[:250])
    elif type(val).__module__.startswith("sklearn"):
        print(key, type(val).__name__)
print("\nRAW EXAMPLES:")
print(train.loc[train.event.isin(["insulin", "norepinephrine"])].head(8).to_string(index=False))
print("\nFORWARD FILES:")
for directory in sorted(Path("/work").glob("fwd_*")):
    print(directory.name, [x.name for x in directory.iterdir()])
```
