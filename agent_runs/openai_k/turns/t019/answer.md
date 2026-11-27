The vocabulary specifies strict bin edges, including repeated insulin edges, so the bin-indexing convention could explain some reconstruction error. The inspection stopped on a printing bug; no weights changed. I’ll fix that inspection and check the helpers. Correcting an indexing error should improve agreement with saved states.

```python
print("=== RETAINED ARRAYS, DICTIONARIES, AND FUNCTIONS ===")
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"key", "value", "items", "text"}:
        continue
    if inspect.isfunction(value):
        try:
            print("FUNCTION", key, inspect.signature(value))
        except Exception:
            print("FUNCTION", key)
    elif isinstance(value, (np.ndarray, pd.DataFrame, pd.Series, torch.Tensor)):
        print("ARRAY", key, type(value).__name__, tuple(value.shape))
    elif isinstance(value, dict):
        print("DICT", key, "keys:", list(value)[:10])
    elif isinstance(value, (list, tuple)):
        print("SEQUENCE", key, "length:", len(value))

print("\n=== DRUG NORMALISATION ===")
for entry in VOCAB["numeric"]:
    if entry["code"] in ("insulin", "norepinephrine"):
        print(entry)

print("\n=== SAVED STATE FILE STRUCTURE ===")
state_path = Path("/work/fwd_3/states_original.npz")
if state_path.exists():
    with np.load(state_path) as sf:
        for key in sf.files:
            arr = sf[key]
            print(key, arr.shape, arr.dtype)
            if arr.ndim <= 1:
                print(" first entries:", arr[:8])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 13–14: vocabulary and retained-state audit\n"
            "- Run 13 printed vocabulary and remaining tensors but its global inventory failed on an imported callable.\n"
            "- Fine-bin edges are strict (edges below x), with repeated insulin edges; reconstruction must honor this.\n"
            "- Run 14 fixes the inventory and inspects saved-state metadata.\n"
            "- Python runs: 14; forward passes: 4; no weights edited.\n")
```