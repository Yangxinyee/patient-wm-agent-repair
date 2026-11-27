The stronger dose carriers barely changed factual fit, but their intervention responses remained too weak. Increasing carrier strength alone therefore looks unpromising. I’ll inspect the latent-to-output pathway and the earlier record-based effect estimates to find where the dose signal is being lost. A usable route should preserve factual predictions while producing the identified response to changed doses.

```python
from pathlib import Path
print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nEARLY NOTES")
notes83 = Path("/work/notes.md").read_text()
print(notes83[:12500])
print("\nRELEVANT KERNEL OBJECTS")
for k in sorted(list(globals())):
    if any(s in k.lower() for s in ("core", "prior", "carrier", "target", "dynamics", "causal", "dose", "impulse", "bridge", "81", "82")):
        v = globals()[k]
        if hasattr(v, "shape"):
            desc = str(tuple(v.shape))
        elif isinstance(v, dict):
            desc = f"dict keys={list(v)[:16]}"
        elif isinstance(v, (list, tuple)):
            desc = f"{type(v).__name__} len={len(v)}"
        else:
            desc = str(type(v))
        print(k, desc)
print("submitted edits:", edited())
```
