The earlier readout attempt failed because it could not reliably decode dose innovations on held-out stays. That rules out repeating the same regression without new evidence. I’ll inspect the newer counterfactual-feature caches and exact head structure to test whether controlled dose contrasts provide a better basis for an edit.

```python
cached_objects = []
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"obj", "tab"}:
        continue
    if isinstance(obj, (pd.DataFrame, np.ndarray, torch.Tensor)):
        cached_objects.append((name, type(obj).__name__, tuple(obj.shape)))
    elif isinstance(obj, dict):
        cached_objects.append((name, "dict", list(obj.keys())[:6]))

print("MOST RECENT CACHES:")
for item in cached_objects[-60:]:
    print(item)

print("\nHEAD/LATENT TENSORS:")
for name in NAMES:
    if any(name.startswith(p) for p in [
        "attr_heads.value", "z_", "last_numeric", "query_norm", "type_emb",
        "gap", "embed_transition.value"
    ]):
        print(name, SHAPES[name])

print("\nHEAD HELPER SOURCE:")
try:
    helper_source = inspect.getsource(reconstruct_reading_head)
    print(helper_source[:3200])
except Exception as exc:
    print(type(exc).__name__, str(exc))
    print("Defined at:", reconstruct_reading_head.__code__.co_filename,
          reconstruct_reading_head.__code__.co_firstlineno)

print("\nPOSTERIOR CONTRAST CV:")
tab_cv = pd.read_csv("/work/posterior_contrast_cv.csv")
print(tab_cv.head(8).to_string(index=False))

print("\nMETADATA EXAMPLES:")
print(head_meta.head(3).to_string(index=False))
print("\nSTATE ARCHIVE KEYS:")
for path in sorted(Path("/work").glob("fwd_*/states_original.npz")):
    archive = np.load(path)
    print(path.parent.name, [(k, archive[k].shape) for k in archive.files])
```