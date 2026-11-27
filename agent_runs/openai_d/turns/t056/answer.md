The records show strong confounding: insulin’s unadjusted association is positive, but becomes about −2 per rate unit after controlling for prior readings. Norepinephrine’s estimate changes substantially with the controls, so its later checks matter. I’ll recover those checks and the saved analysis variables. Stable estimates across held-out stays and control choices would support causal correction; instability would argue for caution.

```python
from pathlib import Path

text = Path("/work/notes.md").read_text()
start = text.find("## Run 4\n")
end = text.find("## Run 8\n")
print("=== CAUSAL CHECKS AND EARLY CACHES ===")
print(text[start:end][:5800])

print("\n=== ANALYSIS VARIABLE INDEX ===")
lines = []
for key in sorted(list(globals())):
    if key.startswith("_") or not any(s in key.lower() for s in
        ("caus", "dml", "dose", "effect", "innov", "panel", "action", "decoder", "gate", "tuning")):
        continue
    value = globals()[key]
    if callable(value):
        summary = "callable"
    elif hasattr(value, "shape"):
        summary = str(tuple(value.shape))
    elif isinstance(value, dict):
        summary = "keys=" + str(list(value)[:12])
    elif isinstance(value, (list, tuple)):
        summary = "length=" + str(len(value))
    else:
        summary = str(value)[:100]
    lines.append(key + ": " + summary)
print("\n".join(lines)[:2900])

print("\n=== ACTION ENCODER TENSORS ===")
for name in NAMES:
    if "encode_action" in name:
        print(name, SHAPES[name])
print("Current edits:", edited())
```
