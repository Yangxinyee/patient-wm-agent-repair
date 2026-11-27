The notebook’s provisional estimates suggest stronger insulin lowering and a positive norepinephrine effect, unlike the original rollouts. They remain hypotheses, not known truth. I’ll now inspect the reconstruction helpers and embedding tables. This should reveal whether the remaining errors come from event encoding rather than the posterior calculation itself.

```python
import inspect, pandas as pd
gcopy = dict(globals())
print("=== FUNCTIONS ===")
for n, v in sorted(gcopy.items()):
    if inspect.isfunction(v) and (
        v.__module__ in {"__main__", "__mp_main__"} or
        n in {"load", "load_original", "save_edit", "drop_edit", "edited"}
    ):
        print(n, str(inspect.signature(v)), "uses:", ",".join(v.__code__.co_names)[:180])
print("\n=== ARRAYS / DATAFRAMES ===")
items = []
for n, v in sorted(gcopy.items()):
    if not n.startswith("_") and hasattr(v, "shape") and not inspect.ismodule(v):
        items.append(f"{n}:{tuple(v.shape)}")
print(" | ".join(items)[:3500])
print("\n=== DICTIONARIES / LONG LISTS ===")
for n, v in sorted(gcopy.items()):
    if n.startswith("_") or n in {"NAMES", "SHAPES", "VOCAB", "gcopy"}:
        continue
    if isinstance(v, dict):
        print(n, len(v), str(list(v)[:6])[:130])
    elif isinstance(v, (list, tuple)) and len(v) > 10:
        print(n, type(v).__name__, len(v), str(v[:2])[:90])
print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
for k in VOCAB:
    if "unit" in k or "categor" in k or "normal" in k:
        print(k, str(VOCAB[k])[:1000])
print("\n=== ENCODER TENSORS ===")
for n in NAMES:
    if n.startswith(("embed.", "encode.", "encode_action.")) or "z_post" in n:
        print(n, SHAPES[n])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 21: compact kernel inventory\n"
            "Recovering helper signatures, array shapes, vocab structure, and encoder tensors "
            "after the previous output was truncated. No edits. Python runs: 21; forward passes: 3.\n")
```
