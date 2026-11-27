The candidate increases value NLL by 0.02689 nats, so it fails the factual-fit requirement. I’ll preserve it for diagnosis and restore the previously tested baseline. Then I’ll recover the architecture and record-based effect estimates to decide whether a smaller edit is worthwhile; only an edit that passes factual validation will be retained.

```python
import os, json, numpy as np, torch
from pathlib import Path

rejected249 = {n: load(n).clone() for n in edited()}
baseline249 = globals().get("safe236")
valid249 = (
    isinstance(baseline249, dict)
    and all(n in SHAPES and hasattr(v, "shape") and tuple(v.shape) == tuple(SHAPES[n])
            for n, v in baseline249.items())
)
for n in list(edited()):
    drop_edit(n)
if valid249:
    for n, v in baseline249.items():
        save_edit(n, v)
print("RESTORED", "safe236" if valid249 else "original", "edited:", edited())

print("\nARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README\n", Path("/data/README.md").read_text())
text249 = Path("/work/notes.md").read_text()
lines249 = text249.splitlines()
hits249 = []
for i, line in enumerate(lines249):
    if any(k in line.lower() for k in (
        "causal", "identif", "instrument", "true effect", "dynamic target",
        "estimated effect", "effect estimate", "residual slope", "distributed",
        "impulse", "dose-response", "dose response"
    )):
        hits249.append((i, "\n".join(lines249[max(0,i-1):min(len(lines249),i+3)])))
print("\nIDENTIFICATION NOTES\n")
for i, s in hits249[-45:]:
    print(f"line {i}: {s}\n")

print("\nRECENT OBJECT INVENTORY")
for n, v in list(globals().items()):
    if n.startswith("_") or not any(k in n.lower() for k in
        ("236", "245", "246", "247", "248", "211", "221", "bmean", "causal", "dyn", "target", "effect")):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(n, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(n, "dict", len(v), "keys", list(v)[:12])
    elif callable(v) and hasattr(v, "__code__"):
        print(n, "function", v.__code__.co_varnames[:v.__code__.co_argcount])
    elif isinstance(v, (str, int, float, list, tuple)):
        print(n, type(v).__name__, str(v)[:240])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 249\nForward24 selective candidate failed factual fit: "
            "1.64565 versus 1.61876, delta +.02689 on 4741 readings. "
            "Preserved as rejected249 and restored "
            + ("safe236" if valid249 else "original") +
            ". Recovering architecture and record-based identification before further edits. "
            "Counts: 249 Python runs, 24 forwards.\n")
```