The architecture confirms that dose inputs affect the shared recurrent state, so broad input edits can disturb every prediction head. The recovered estimates also vary substantially with the controls used, making simple associations unreliable causal targets. I’ll recover the identification checks and cached objects next; agreement across independent checks would support a repair target.

```python
print("=== IDENTIFICATION NOTES, NEXT SECTION ===")
print(notes_text[3800:9500])

print("\n=== SAVED CAUSAL TABLES ===")
for filename in ["dynamic_iv.csv", "proxy_iv.csv", "sensor_iv.csv", "kernel_targets.csv"]:
    frame = pd.read_csv("/work/" + filename)
    print(filename, "shape", frame.shape, "columns", frame.columns.tolist())
    print(frame.head(2).to_string(index=False))

print("\n=== CACHED DATAFRAMES AND ARRAYS ===")
objects = [(n, globals()[n]) for n in sorted(list(globals())) if not n.startswith("_")]
print([(n, tuple(o.shape)) for n, o in objects
       if isinstance(o, (pd.DataFrame, np.ndarray))])
print("\n=== CACHED FUNCTIONS ===")
print([(n, o.__code__.co_varnames[:o.__code__.co_argcount])
       for n, o in objects
       if callable(o) and getattr(o, "__module__", "") == "__main__"
       and hasattr(o, "__code__")])
print("Notebook length:", len(notes_text), "Active edits:", edited())
```
