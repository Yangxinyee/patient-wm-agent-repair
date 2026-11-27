The rollout refutes simple attention-gain calibration: reducing the insulin payload made its response more negative, and halving insulin even moved glucose in the wrong direction. Norepinephrine improved, but factual fit still failed. I’ll restore the validated checkpoint and inspect the transport–compensation coupling; a useful repair must preserve the factual baseline while producing opposite-signed dose responses.

```python
import re
from pathlib import Path
import numpy as np
import torch

checkpoint250 = {n: load(n).clone() for n in edited()}
assert isinstance(checkpoint249, dict)
assert checkpoint249 and all(n in NAMES for n in checkpoint249)
for n in list(edited()):
    if n not in checkpoint249:
        drop_edit(n)
for n, v in checkpoint249.items():
    save_edit(n, v)
print("Restored validated checkpoint222:",
      all(torch.equal(load(n), torch.as_tensor(v, dtype=torch.float32))
          for n, v in checkpoint249.items()))

text252 = Path("/work/notes.md").read_text()
entries252 = text252.split("\n\n")
print("=== DESIGN / IDENTIFICATION NOTES ===")
selected252 = [
    e for e in entries252
    if re.search(r"Python runs?: (1[0-9]|18[0-9]|19[0-9]|20[0-9])(?:[;., ])", e)
]
for e in selected252:
    if any(k in e.lower() for k in
           ("dynamic", "innovation", "carrier", "compensation",
            "checkpoint193", "direction", "gate", "linear")):
        print(e[:650])
        if sum(len(x[:650]) for x in selected252[:selected252.index(e)+1]) > 4300:
            break

print("=== CALLABLE HELPERS ===")
helpers252 = [
    n for n, v in list(globals().items())
    if callable(v) and hasattr(v, "__code__") and not n.startswith("_")
]
print(helpers252)

print("=== FIT / TRANSPORT VARIABLES ===")
inventory252 = []
for n, v in list(globals().items()):
    if re.search(r"(193|206|220|222|243|249|250)$", n):
        if hasattr(v, "shape"):
            inventory252.append((n, type(v).__name__, tuple(v.shape)))
        elif isinstance(v, dict):
            inventory252.append((n, "dict", list(v)[:4]))
        elif isinstance(v, (int, float, str)):
            inventory252.append((n, type(v).__name__, str(v)[:70]))
print(inventory252[:65])

with Path("/work/notes.md").open("a") as log252:
    log252.write(
        "\nPython run 252; forwards 20. Candidate250 rejected: factual "
        "+0.00818 nats; rollouts (-20.393,-1.319,+2.027,-2.226). "
        "Insulin response does not follow attention-payload gain. Restored "
        "validated checkpoint222 from checkpoint249. Inspecting joint "
        "transport/compensation before further edits.\n"
    )
```