The architecture confirms that predictions combine recurrent state with a same-code “last reading” register. A register mismatch could explain the failed reconstruction, but the output was truncated before showing the relevant objects. I’ll recover those details with bounded output; a documented convention that reproduces GPU predictions would support this hypothesis.

```python
notes = Path("/work/notes.md").read_text()
print("=== EARLY NOTES ===")
print(notes[:11500])

print("\n=== COMPACT OBJECT INVENTORY ===")
for nm, ob in sorted(list(globals().items())):
    if nm.startswith("_") or nm in {"W", "WO", "BASE217", "NAMES", "SHAPES", "VOCAB", "notes"}:
        continue
    if inspect.isfunction(ob):
        print(nm, "function", list(inspect.signature(ob).parameters))
    elif isinstance(ob, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(nm, type(ob).__name__, tuple(ob.shape))
    elif isinstance(ob, dict):
        print(nm, "dict", len(ob), repr(list(ob)[:12]))
    elif isinstance(ob, (list, tuple)):
        print(nm, type(ob).__name__, len(ob))
    elif isinstance(ob, (str, int, float, np.integer, np.floating)):
        print(nm, repr(ob)[:100])

print("\n=== VOCABULARY CONFIG AND NUMERIC LAYOUT ===")
print(V["config"])
print(V["numeric_note"])
print(V["numeric"]["glucose_finger"] if "glucose_finger" in V["numeric"] else V["numeric"])
print("\n=== LAST-NUMERIC ORIGINAL AND EDITED COLUMN NORMS ===")
for label, ww in [("original", WO), ("edited", W)]:
    print(label, torch.linalg.vector_norm(ww["last_numeric.weight"], dim=0).tolist())
```
