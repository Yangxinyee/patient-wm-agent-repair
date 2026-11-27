The factual test passed: value NLL increased by 0.00229 nats, below the 0.005 limit. But the rollout shows that these channels are not producing the needed drug-specific effects. I’ll recover the identification evidence and channel construction; this should distinguish a bad direction from insufficient strength.

```python
from pathlib import Path
import inspect

notes = Path("/work/notes.md").read_text()
print("=== INITIAL IDENTIFICATION EVIDENCE ===")
print(notes[:6500])

print("\n=== CANDIDATE AND CALIBRATION VARIABLES ===")
wanted = ("candidate", "direction", "channel", "recipe", "impulse",
          "causal", "decoder", "target", "calib")
items = []
for k, v in list(globals().items()):
    if any(s in k.lower() for s in wanted) and not inspect.isfunction(v):
        shape = str(v.shape) if hasattr(v, "shape") else ""
        detail = ""
        if isinstance(v, dict):
            detail = " keys=" + str(list(v)[:12])
        elif isinstance(v, (list, tuple)):
            detail = " len=" + str(len(v))
        items.append(f"{k}: {type(v).__name__} {shape}{detail}")
print("\n".join(items)[:1900])

print("\n=== LATER FUNCTIONS ===")
fs = [(k, v) for k, v in list(globals().items())
      if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__"]
start = next((i for i, (k, _) in enumerate(fs) if k == "set_post71"), 0)
for k, v in fs[start:]:
    try:
        print(k, str(inspect.signature(v))[:150])
    except Exception:
        print(k)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 141: factual margin and identification recovery\n"
            "- Forward 8 value NLL delta: +0.002286 nats/reading on tuning-99; within constraint.\n"
            "- Current rollout remains far short of identification-based drug targets, especially NE.\n"
            "- Recovering empirical evidence and correction-channel construction before redesign.\n"
            "- Python runs: 141; forward passes: 9.\n")
```