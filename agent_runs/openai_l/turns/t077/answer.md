The retained estimates support an immediate insulin effect near −1.95 and a norepinephrine effect near +3.1; the longer-lag effects remain less certain. I’ll test the previously validated action encoder directly. A near-zero reconstruction error would let us measure dose-induced state changes without relying on the failed encoder.

```python
import numpy as np
import torch
import inspect
import dis
from pathlib import Path

print("=== TRUSTED GENERIC ACTION ENCODER ===")
emb70 = atom_embedding49(
    drugs65, rates65, doses65, kind=0, ekind=2,
    units="native", attributes="full", action_category=0
)
u70 = encode_set50(
    emb70, owners65, counts65, "encode_action", zero_if_empty=False
)
print("embedding RMS versus emb65:",
      float(torch.mean((emb70 - emb65) ** 2).sqrt()))
print("pooled RMS versus validated u:",
      float(torch.mean((u70 - u_valid60) ** 2).sqrt()))
print("failed explicit encoder RMS versus generic:",
      float(torch.mean((emb67 - emb70) ** 2).sqrt()))

print("\n=== STATE/RECURRENCE OBJECTS ===")
lines70 = []
for n70, o70 in list(globals().items()):
    if (n70.endswith(("60", "63", "64"))
        or (hasattr(o70, "shape") and len(o70.shape) == 2
            and o70.shape[-1] == 992)):
        if hasattr(o70, "shape"):
            lines70.append(f"{n70}: {type(o70).__name__} {tuple(o70.shape)}")
        elif isinstance(o70, dict):
            lines70.append(f"{n70}: keys={list(o70)[:15]}")
        elif isinstance(o70, (str, int, float, bool)):
            lines70.append(f"{n70}: {repr(o70)[:100]}")
print("\n".join(lines70)[:1900])

print("\n=== CONTEXT FUNCTION ===")
try:
    print(inspect.getsource(context64))
except Exception:
    print(dis.Bytecode(context64).dis()[:2500])

print("\n=== DECODER FUNCTION ===")
try:
    print(inspect.getsource(decode67))
except Exception:
    print(dis.Bytecode(decode67).dis()[:3500])

with Path("/work/notes.md").open("a") as f70:
    f70.write(
        "\n## Run 70\n"
        "Accepted immediate effects remain I=-1.95 and NE=+3.1, with longer-lag "
        "uncertainty. Testing the previously validated generic action encoder "
        "instead of the failed explicit encoder before any intervention edit.\n"
    )
```