The zero-payload transport edit preserved factual fit, and the retained decoder estimates recorded doses accurately. That makes it a promising route, but it has not yet demonstrated a drug effect. I’ll recover its implementation and the record-derived effect estimates before adding a signal. The hypothesis will hold only if actual rollouts improve while factual NLL remains within budget.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== CURRENT EDITS ===")
for name in edited():
    w = load(name)
    w0 = load_original(name)
    print(name, tuple(w.shape),
          "delta_norm", float(np.linalg.norm(np.asarray(w) - np.asarray(w0))))

print("\n=== RETAINED FUNCTIONS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and not name.startswith("_"):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name, sig)

print("\n=== RECENT TRANSPORT / DECODER VARIABLES ===")
for name, obj in list(globals().items()):
    if any(s in name.lower() for s in
           ("103", "104", "105", "145", "146", "147", "148",
            "transport", "payload", "policy", "direction", "causal", "kernel")):
        if isinstance(obj, np.ndarray):
            print(name, obj.shape, obj.dtype)
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj.keys())[:35])
        elif isinstance(obj, (str, int, float, bool, list, tuple)):
            text = repr(obj)
            print(name, text[:1800])

print("\n=== EARLY RECORD-IDENTIFICATION NOTES ===")
notes149 = Path("/work/notes.md").read_text()
lines149 = notes149.splitlines()
selected149 = [
    i for i, line in enumerate(lines149)
    if any(s in line.lower() for s in
           ("instrument", "random", "confound", "identif", "impulse",
            "kernel", "reference", "causal", "30,000", "30000"))
]
shown149 = set()
for i in selected149:
    for j in range(max(0, i - 1), min(len(lines149), i + 2)):
        if j not in shown149 and len(shown149) < 100:
            print(lines149[j])
            shown149.add(j)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\nPython run 149; forwards 10. Restart recovery: zero-payload "
        "latent-head transport remains installed and factual-neutral "
        "(+0.00003 NLL). Retained action decoder rate RMSE is about "
        "0.021 insulin and 0.016 NE; context-policy prediction errors "
        "are much larger. Recovering exact transport, decoder, and "
        "record-identification evidence before adding any dose payload. "
        "No weights changed in this run.\n"
    )
```