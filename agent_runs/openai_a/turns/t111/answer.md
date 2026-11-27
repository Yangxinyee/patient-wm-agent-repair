The posterior reconstruction now matches the saved states to numerical precision. That resolves the encoding ambiguity, but the full state update still needs validation. I’ll recover the architecture and earlier causal evidence before constructing synthetic dose contrasts. Matching saved next states would support this approach; a mismatch would block it.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

r97_notes = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK EVIDENCE ===")
print(r97_notes[:22000])

print("\n=== KERNEL INVENTORY ===")
for r97_name, r97_obj in sorted(list(globals().items())):
    if r97_name.startswith("_") or r97_name.startswith("r97_"):
        continue
    if callable(r97_obj):
        try:
            r97_desc = str(inspect.signature(r97_obj))
        except Exception:
            r97_desc = type(r97_obj).__name__
    elif hasattr(r97_obj, "shape"):
        r97_desc = f"{type(r97_obj).__name__} shape={tuple(r97_obj.shape)}"
    elif isinstance(r97_obj, dict):
        r97_desc = f"dict len={len(r97_obj)} keys={list(r97_obj)[:10]}"
    elif isinstance(r97_obj, (list, tuple)):
        r97_desc = f"{type(r97_obj).__name__} len={len(r97_obj)}"
    elif isinstance(r97_obj, (str, int, float, bool)):
        r97_desc = repr(r97_obj)[:160]
    else:
        r97_desc = type(r97_obj).__name__
    print(r97_name, ":", r97_desc)

with Path("/work/notes.md").open("a") as r97_f:
    r97_f.write(
        "\n## Restart recovery; Python 97, forward 11/50\n"
        "- Posterior reconstruction reached RMSE 2.0e-7; full state update remains unvalidated.\n"
        "- Recovering architecture, causal evidence, and retained kernel objects before synthetic contrasts.\n"
        "- No new weight edits; validated beta=-2 insulin graft remains active.\n"
    )
```
