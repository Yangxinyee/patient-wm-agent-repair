The two action encoders agree to numerical precision, so the large error comes from pooling or bundle alignment, not dose embedding. I’ll separate empty from action-bearing bundles and inspect the pooling logic. Matching the reconstructed posterior means would confirm that the action path is usable.

```python
import numpy as np
import torch
import dis
from pathlib import Path

print("=== POOLING ERROR BY BUNDLE TYPE ===")
for label71, mask71 in [
    ("empty", np.asarray(counts65) == 0),
    ("one action", np.asarray(counts65) == 1),
    ("multiple actions", np.asarray(counts65) > 1),
]:
    if np.any(mask71):
        mt71 = torch.as_tensor(mask71)
        print(label71, "n=", int(mask71.sum()),
              "RMS=", float((u70[mt71] - u_valid60[mt71]).square().mean().sqrt()),
              "generic norm=", float(u70[mt71].square().mean().sqrt()),
              "reference norm=", float(u_valid60[mt71].square().mean().sqrt()))
print("u65 versus u70 RMS:", float((u65-u70).square().mean().sqrt()))
print("count distribution:", np.unique(counts65, return_counts=True))
print("owner range:", np.min(owners65), np.max(owners65))
print("first active rows:", np.flatnonzero(np.asarray(counts65) > 0)[:12])
print("first generic/reference coordinates:", u70[:3, :4], u_valid60[:3, :4])

def compact_bytecode71(f, limit=2200):
    return "\n".join(
        f"{i.offset}: {i.opname} {i.argrepr}"
        for i in dis.get_instructions(f)
    )[:limit]

print("\n=== GENERIC POOL FUNCTION ===")
print(compact_bytecode71(encode_set50, 2300))
print("\n=== ACTION POOL FUNCTION ===")
print(compact_bytecode71(pool_actions49, 1800))

print("\n=== STATE/ALIGNMENT OBJECTS ===")
lines71 = []
for n71, o71 in list(globals().items()):
    if isinstance(o71, (np.ndarray, torch.Tensor)):
        sh71 = tuple(o71.shape)
        if ((len(sh71) == 2 and sh71[-1] == 992)
            or n71.endswith(("49", "50", "53", "55", "60", "63", "64"))):
            lines71.append(f"{n71}: {type(o71).__name__} {sh71}")
    elif isinstance(o71, dict) and n71.endswith(("49", "50", "53", "55", "60", "63", "64")):
        lines71.append(f"{n71}: keys={list(o71)[:12]}")
print("\n".join(lines71)[:2200])

with Path("/work/notes.md").open("a") as f71:
    f71.write(
        "\n## Runs 70–71 / pooling audit\n"
        "Explicit and trusted generic atom embeddings agree to 3.1e-8, but pooled "
        "u differs from u_valid60 by RMS 1.0097. Run 70 stopped at a metadata "
        "inspection error. Localizing pooling versus bundle alignment; no edits.\n"
    )
```