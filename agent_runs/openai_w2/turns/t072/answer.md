```python
print(open("/model/ARCHITECTURE.md").read())
print("\n--- DATA README ---\n", open("/data/README.md").read())
print("\n--- FULL NOTEBOOK ---\n", open("/work/notes.md").read())
print("\n--- KERNEL VARIABLES ---")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or k in {"In", "Out"}:
        continue
    if hasattr(v, "shape"):
        print(k, type(v).__name__, v.shape)
    elif isinstance(v, (dict, list, tuple)):
        print(k, type(v).__name__, "len", len(v), "sample", repr(v)[:350])
    elif callable(v):
        print(k, "callable")
print("\nCURRENT EDITS:", edited())
```
