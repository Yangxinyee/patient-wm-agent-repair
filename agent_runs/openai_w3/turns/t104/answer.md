The last diagnostic was truncated before the useful helpers and identification results appeared. No weights changed. I’ll retrieve only those pieces now so the next step can test a small action-path edit rather than repeat broad inspection.

```python
print("REPLAY/READOUT FUNCTIONS")
for n in sorted(list(globals())):
    o = globals()[n]
    if inspect.isfunction(o) and any(k in n.lower() for k in ("step", "head", "value", "prelude", "replay", "readout", "block", "predict", "kernel")):
        print(n, inspect.signature(o), "uses", o.__code__.co_names)

print("\nRECENT GLOBAL NAMES")
print(" ".join(n for n in sorted(list(globals()))
               if not n.startswith("_") and re.search(r"(8[2-9]|90)$", n)))

print("\nKEY CONTAINERS")
for n in ("S", "W", "r40_info", "registers90", "fits", "causal", "targets",
          "results", "true_targets", "tuning", "train"):
    if n in globals():
        o = globals()[n]
        if isinstance(o, dict):
            print(n, "dict", len(o), "first keys", list(o)[:15])
        elif hasattr(o, "shape"):
            print(n, type(o).__name__, o.shape,
                  "columns" if hasattr(o, "columns") else "",
                  list(o.columns) if hasattr(o, "columns") else "")
        else:
            print(n, type(o).__name__)

print("\nIDENTIFICATION NOTEBOOK")
print(notes92[:5500])

print("\nPRELUDE CACHE")
try:
    print(inspect.getsource(prelude_cache88))
except Exception:
    print(compact_bytecode(prelude_cache88))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 93; forwards 7. Run 92 output was truncated; no edits. "
            "Retrieving only replay/readout functions and record-identification summary.\n")
```