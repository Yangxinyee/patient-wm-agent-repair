The architecture confirms that drug doses enter through a separate action encoder, but the object listing was truncated before the useful diagnostics appeared. No weights have changed. I’ll retrieve a compact inventory and the earlier causal evidence; consistent effects across independent diagnostics would support a repair target, while disagreement would mean we need more identification work.

```python
from pathlib import Path

print("DATAFRAMES:")
print([(n, tuple(v.shape)) for n, v in list(globals().items())
       if not n.startswith("_") and isinstance(v, pd.DataFrame)])
print("\nFUNCTIONS:")
print([n for n, v in list(globals().items())
       if callable(v) and getattr(v, "__module__", "") == "__main__"])
print("\nSMALL DICTIONARY INVENTORY:")
for n, v in sorted(list(globals().items())):
    if isinstance(v, dict) and n not in ("SHAPES", "VOCAB", "W_repair", "__builtins__"):
        print(n, len(v), repr(list(v.keys())[:6]))
print("\nEARLY NOTEBOOK:")
print(Path("/work/notes.md").read_text()[:4800])
print("\nATOM FUNCTION METADATA:")
for n in ("atom_embedding_repair", "act_repair", "build_head_dataset_fixed",
          "build_history_features"):
    fn = globals().get(n)
    if callable(fn):
        print(n, "arguments:", fn.__code__.co_varnames[:fn.__code__.co_argcount],
              "referenced names:", fn.__code__.co_names)
print("\nCORE TRANSFORMER TENSORS:")
for n in NAMES:
    if any(s in n for s in ("prelude.", "blocks.0.0.", "coda.", "last_status")):
        print(n, SHAPES[n])
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 54\nArchitecture recovered; run-53 object output truncated. Retrieving compact causal/reconstruction inventory rather than repeating the full inventory. No weights edited.\n")
```