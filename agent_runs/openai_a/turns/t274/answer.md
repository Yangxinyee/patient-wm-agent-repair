The activity-gated edit increases factual NLL by 0.00487 nats on these 100 stays—barely within the limit. That does not yet establish safety or better dose responses. I’ll recover the saved construction and record-derived targets before testing another cohort; those tests could confirm the gate helps, or show that we need to restore the safer checkpoint.

```python
from pathlib import Path
print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())
notes_text = Path("/work/notes.md").read_text()
print("\n=== NOTEBOOK: EARLY EVIDENCE ===")
print(notes_text[:15000])
print("\n=== KERNEL INVENTORY ===")
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    v = globals()[k]
    if callable(v):
        if any(s in k.lower() for s in ["head", "gate", "cache", "install", "loss", "checkpoint", "build", "predict", "effect"]):
            print(k, "CALLABLE")
    elif any(s in k.lower() for s in ["checkpoint", "tuning", "ids", "activity", "r3", "finger", "target"]):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict keys:", list(v.keys())[:35])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, "len", len(v), "first", v[:6])
        elif isinstance(v, (str, int, float, bool)):
            print(k, repr(v)[:300])
print("\nCURRENT EDITS:", edited())
```
