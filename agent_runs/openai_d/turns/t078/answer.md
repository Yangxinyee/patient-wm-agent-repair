This repeats the architecture output rather than reporting a new experiment. No edit has been tested or installed. I’ll request a shorter recovery output so the refined causal estimates and recent cached variables fit without truncation. Those estimates will determine which dose responses need correcting.

```python
from pathlib import Path

rec70_text = Path("/work/notes.md").read_text()
rec70_start = rec70_text.find("## Run 4\n")
print("=== CAUSAL ESTIMATES AND EARLY EXPERIMENTS ===")
print(rec70_text[rec70_start:rec70_start + 4800])

rec70_vars = list(globals().items())
rec70_arrays = [
    f"{n}:{tuple(v.shape)}"
    for n, v in reversed(rec70_vars)
    if not n.startswith(("_", "rec")) and hasattr(v, "shape")
]
rec70_dicts = [
    f"{n}({len(v)}):{list(v.keys())[:4]}"
    for n, v in reversed(rec70_vars)
    if not n.startswith(("_", "rec"))
    and isinstance(v, dict)
    and n not in ("SHAPES", "VOCAB")
]
rec70_funcs = [
    n for n, v in rec70_vars
    if callable(v) and getattr(v, "__module__", "") == "__main__"
    and not n.startswith("_")
]

print("\n=== RECENT ARRAYS ===")
print("; ".join(rec70_arrays)[:2400])
print("\n=== RECENT DICTIONARIES ===")
print("; ".join(rec70_dicts)[:1400])
print("\n=== LOCAL FUNCTIONS ===")
print(", ".join(rec70_funcs)[:1000])
print("\nCurrent edits:", edited())
```