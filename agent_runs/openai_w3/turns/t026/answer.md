The architecture confirms that doses affect both the latent state and recurrent dynamics; the value head also uses the previous reading. The output was cut off before showing the existing helpers. I’ll inspect those and the embedding metadata next, so we can isolate the reconstruction error without changing weights.

```python
import inspect, json

print("=== LIVE FUNCTIONS ===")
for key, obj in list(globals().items()):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(key, inspect.signature(obj))
        except Exception:
            print(key)

print("\n=== LIVE ARRAYS / DATAFRAMES ===")
for key, obj in list(globals().items()):
    if key.startswith("_"):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(key, tuple(obj.shape))
    elif hasattr(obj, "columns") and hasattr(obj, "shape"):
        print(key, obj.shape, list(obj.columns))

print("\n=== LIVE DICTIONARIES ===")
for key, obj in list(globals().items()):
    if isinstance(obj, dict) and not key.startswith("_") and key not in ("SHAPES",):
        print(key, len(obj), str(list(obj.keys())[:10])[:250])

print("\n=== VOCAB METADATA ===")
for key in ("config", "codes", "event_kinds", "atom_kinds", "numeric_note"):
    print(key, VOCAB[key])
print("numeric", json.dumps(VOCAB["numeric"])[:3200])

print("\n=== EMBEDDING TENSORS ===")
for name in NAMES:
    if name.startswith("embed.") and not name.startswith("embed_transition"):
        print(name, SHAPES[name])

print("\n=== HISTORY AVAILABILITY ===")
print("In" in globals(), "get_ipython" in globals())
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 22; forwards 3. Architecture recovered; previous output truncated before live helpers. Inspecting compact metadata and helpers. No edits.\n")
```