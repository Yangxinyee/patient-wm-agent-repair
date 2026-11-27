The architecture confirms that doses enter a shared recurrent state, so a state-level correction could affect both sensors and later hours. The record analyses support effects near −2 for insulin and +3 for norepinephrine per rate unit, but those are next-reading effects—not rollout totals. I’ll inspect the bundle inputs and retained lag estimates; this should reveal whether the reconstruction error is a correctable encoding detail or makes that editing route unsafe.

```python
import json, pandas as pd, torch
from pathlib import Path

print("RECORD SCHEMA\n", Path("/data/README.md").read_text()[:3200])
print("\nSTATE FILES")
for p102 in sorted(Path("/work").glob("fwd_*/*")):
    if p102.suffix == ".npz":
        with np.load(p102) as z102:
            print(str(p102), [(k, z102[k].shape) for k in z102.files])

print("\nRELEVANT RETAINED OBJECTS")
lines102 = []
for n102 in sorted(list(globals())):
    if n102.startswith("_"):
        continue
    o102 = globals()[n102]
    if any(s in n102.lower() for s in
           ("panel", "case", "state", "bundle", "lag", "impulse", "target", "innovation")):
        if hasattr(o102, "shape"):
            lines102.append(f"{n102}: {type(o102).__name__} {o102.shape}")
        elif isinstance(o102, dict):
            lines102.append(f"{n102}: dict[{len(o102)}] {list(o102)[:5]}")
        elif isinstance(o102, (list, tuple)):
            lines102.append(f"{n102}: {type(o102).__name__}[{len(o102)}]")
        elif callable(o102):
            try:
                lines102.append(f"{n102}{inspect.signature(o102)}")
            except Exception:
                pass
print("\n".join(lines102[:65]))

print("\nLAG-EVIDENCE EXCERPTS")
pos102 = [m.start() for m in __import__("re").finditer(
    "convolution|impulse|persistence", notes101, flags=__import__("re").I)]
for i102 in pos102[:3]:
    print(notes101[max(0, i102-180):i102+1000])

v102 = json.loads(Path("/model/vocab.json").read_text())
print("\nVOCAB KEYS", list(v102))
print("CATEGORICAL VOCAB", {k:v for k,v in v102.items()
                              if any(s in k.lower() for s in ("categor", "unit", "kind"))})
print("EDITED", edited())

with Path("/work/notes.md").open("a") as f102:
    f102.write("\n## Python 102 — shared-state correction prerequisites\n"
               "- Architecture confirms separate action encoding into a shared recurrent state.\n"
               "- Proximal record effects cannot be substituted for accumulated rollout effects.\n"
               "- Inspecting bundle schema, retained lag evidence, and state files; no weights changed.\n")
```
