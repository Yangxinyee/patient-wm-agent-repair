The architecture confirms that a value-head-only edit leaves the recurrent state machinery unchanged, which makes it a relatively contained intervention. The output was truncated before the cache details, so I’ll inspect those directly. I need a verified local loss calculation before choosing another candidate.

```python
print("=== CACHE AND EVALUATOR INVENTORY ===")
for name in sorted(list(globals())):
    if name.startswith(("r203", "r207", "r208", "r209", "r210", "r211")):
        v = globals()[name]
        if callable(v):
            try:
                desc = str(inspect.signature(v))
            except Exception:
                desc = type(v).__name__
            print(name, desc)
        elif isinstance(v, dict):
            print(name, "keys", list(v)[:20])
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            print(name, tuple(v.shape), str(v.dtype))
        elif hasattr(v, "shape"):
            print(name, type(v).__name__, v.shape)
        else:
            print(name, type(v).__name__, repr(v)[:140])

print("=== FOUR-TENSOR BACKUPS ===")
for name in ["r170_insulin_backup", "r171_combined_backup", "c183_fallback",
             "c185_current", "c196_strong_backup"]:
    if name in globals():
        v = globals()[name]
        print(name, {k: (tuple(x.shape), type(x).__name__) for k, x in v.items()}
              if isinstance(v, dict) else type(v).__name__)

print("=== CAUSAL NOTEBOOK ENTRIES ===")
r213_text = Path("/work/notes.md").read_text()
r213_sections = r213_text.split("\n## ")
r213_found = 0
for section in r213_sections:
    if any(s in section.lower() for s in
           ["causal identification", "cross-fitted", "persistence", "dose exogeneity"]):
        print(section[:950])
        r213_found += 1
        if r213_found >= 4:
            break
print("=== EDITED ===", edited())
```