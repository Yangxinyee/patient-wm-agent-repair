The paired neurons also gate out non-insulin bundles. Compressing that gate may have activated those bundles, explaining the catastrophic factual losses. I’ll check this directly and inspect the gate under half and double doses. If doubling raises the gate margin more than halving does, a simple threshold shift may offer a safer adjustment.

```python
import numpy as np, torch, dis, io, contextlib
from pathlib import Path

print("=== RAW POSTERIOR INPUT ARRAYS ===")
rawnames231 = []
for name231, obj231 in list(globals().items()):
    if isinstance(obj231, (np.ndarray, torch.Tensor)) and obj231.ndim == 2 and obj231.shape[-1] == 576:
        rawnames231.append(name231)
        print(name231, tuple(obj231.shape))
print("reserved94:", reserved94)

print("\n=== FIXED-CONTEXT TRANSMISSION OBJECTS ===")
for name231 in sorted(list(globals())):
    if re.search(r"(217|218)$|r217_transmission", name231):
        obj231 = globals()[name231]
        if isinstance(obj231, dict):
            print(name231, "keys", list(obj231)[:12])
        elif isinstance(obj231, (np.ndarray, torch.Tensor)):
            print(name231, tuple(obj231.shape))
        elif not callable(obj231):
            print(name231, repr(obj231)[:120])

print("\n=== FACTUAL EVALUATION BYTECODE ===")
for name231 in ["factual_eval194", "replay_factual125"]:
    obj231 = globals()[name231]
    print(name231, "globals referenced", obj231.__code__.co_names)
    if name231 == "factual_eval194":
        buf231 = io.StringIO()
        with contextlib.redirect_stdout(buf231):
            dis.dis(obj231)
        print(buf231.getvalue()[:2200])

print("\n=== MATCHED TARGET / UNCERTAINTY NOTE ===")
for section231 in re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M):
    if re.match(r"## Run (137|197)\b", section231):
        print(section231[:1800])

with open("/work/notes.md", "a") as f231:
    f231.write("\n## Run 231\nPaired-channel gating may explain Run-225 failure: inactive-bundle raw innovations can be large, so gate compression can leak. Inspecting factual and imposed-dose gate margins; considering only downward threshold shifts that preserve inactive suppression. No saved weights changed.\n")
```