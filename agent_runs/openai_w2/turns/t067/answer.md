The backup command failed because the weights are PyTorch tensors, so no weights were changed or reverted. I’ll retry with tensor cloning, then inspect states and decoder shapes. The proposed residual-direction edit remains only a hypothesis.

```python
import numpy as np, pandas as pd, torch
from pathlib import Path

bin4_backup = {n: load(n).clone() for n in edited()}
for n in list(edited()):
    drop_edit(n)
print("CURRENT EDITS:", edited())

print("\nRELEVANT TENSORS:")
for n in NAMES:
    if any(s in n for s in ["attr_heads.value", "last_numeric", "query_norm",
                             "encode_action", "core.prelude", "core.out_norm",
                             "z_head", "z_post", "core.c_out"]):
        print(n, SHAPES[n])

print("\nKERNEL OBJECTS:")
objects = []
for key, val in list(globals().items()):
    if key.startswith("_") or key in {"In", "Out", "val", "notes_text", "bin4_backup"}:
        continue
    if isinstance(val, (pd.DataFrame, np.ndarray, torch.Tensor)):
        objects.append((key, type(val).__name__, tuple(val.shape)))
    elif callable(val) and getattr(val, "__module__", "") == "__main__":
        objects.append((key, "function"))
    elif isinstance(val, dict) and key not in {"SHAPES", "VOCAB"}:
        objects.append((key, "dict", list(val.keys())[:6]))
for row in objects[:90]:
    print(row)

print("\nSAVED STATE FILES:")
for p in sorted(Path("/work").glob("fwd_*/*.npz")):
    with np.load(p) as zf:
        print(str(p), [(k, zf[k].shape) for k in zf.files])

notes_text = Path("/work/notes.md").read_text()
print("\nNOTEBOOK SECTION HEADINGS:")
print("\n".join(x for x in notes_text.splitlines() if x.startswith("##")))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 52–54; forwards 11–12: reject bin amplification\n"
            "Dose/rate bin x4 with mean code compensation raised factual NLL by "
            "0.00910 (fails budget). Rollout insulin x2/half -3.032/+1.511; "
            "NE x2/half -0.839/-1.798. Reverted all edits. Simple action-embedding "
            "amplification is inadequate; investigating a conditional-dose residual "
            "direction in original states for a targeted analytical edit.\n")
```
