The probe changed the recurrent slots substantially, but changed the normalized output only slightly. Raw state differences therefore may overstate what the prediction head can detect. The diagnostic edit is removed. Next I’ll recover the head-reconstruction helpers and test whether the probe produces a usable head-input direction without introducing large factual changes.

```python
def r126_desc(v):
    if isinstance(v, dict):
        return {"type": "dict", "keys": list(v)[:30]}
    if hasattr(v, "shape"):
        return {"type": type(v).__name__, "shape": tuple(v.shape)}
    if isinstance(v, (list, tuple)):
        return {"type": type(v).__name__, "length": len(v),
                "preview": [str(x)[:70] for x in v[:5]]}
    if isinstance(v, (str, int, float, bool, np.number)):
        return repr(v)[:200]
    return type(v).__name__

print("--- Recent variables ---")
for k, v in sorted(list(globals().items())):
    if k.startswith(("r124", "r123", "r122", "r121", "r120",
                     "r119", "r118", "r117", "r115")):
        print(k, r126_desc(v))

print("\n--- Helper names and parameter names (no defaults) ---")
for k, v in sorted(list(globals().items())):
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        try:
            print(k, list(inspect.signature(v).parameters))
        except (TypeError, ValueError):
            print(k)

print("\n--- DataFrames ---")
for k, v in sorted(list(globals().items())):
    if isinstance(v, pd.DataFrame):
        print(k, v.shape, list(v.columns)[:24])

print("\n--- Relevant saved objects ---")
terms = ("head", "gate", "kernel", "innovation", "latent", "draw", "causal")
for k, v in sorted(list(globals().items())):
    if any(t in k.lower() for t in terms) and not callable(v):
        print(k, r126_desc(v))

print("\n--- Vocabulary structure ---")
print({k: r126_desc(v) for k, v in VOCAB.items()})
print("\n--- Earlier notebook evidence ---")
r126_notes = open("/work/notes.md").read()
print(r126_notes[:8500])
```