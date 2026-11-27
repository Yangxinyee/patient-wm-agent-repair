Baseline77 also fails on the fresh sample: its NLL increase is 0.00592, above the 0.005 limit. I’ll recover the record-derived effect targets and earlier trade-offs before adjusting it. Shrinkage is promising only if it creates factual margin without losing too much of the causal correction.

```python
from pathlib import Path
import re, json, numpy as np, pandas as pd, torch

print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README\n", Path("/data/README.md").read_text())

notes233 = Path("/work/notes.md").read_text()
sections233 = re.split(r"(?=^## )", notes233, flags=re.M)
print("\nEARLY RECORD-BASED EVIDENCE\n", notes233[:10500])

wanted233 = {159, 160, 163, 166, 169, 177, 178, 185, 186, 189,
             199, 205, 208, 211, 213, 216, 219, 220, 221}
print("\nSELECTED LATER RESULTS")
for sec233 in sections233:
    heading233 = sec233.splitlines()[0] if sec233.splitlines() else ""
    nums233 = {int(x) for x in re.findall(r"\b(?:Run|Runs)\s+(\d+)", heading233)}
    if nums233 & wanted233:
        print(sec233[:2200])

print("\nSAVED CANDIDATE CONTRACTS")
for file233 in ["baseline77.npz", "insulin_trial219.npz",
                "ne_candidate151.npz", "ne_candidate199.npz",
                "ne_candidate155.npz"]:
    with np.load("/work/" + file233, allow_pickle=False) as archive233:
        print(file233, {k: (archive233[k].shape, str(archive233[k].dtype))
                        for k in archive233.files})

print("\nRELEVANT PERSISTENT VARIABLES")
snapshot233 = list(globals().items())
selected233 = [(k, v) for k, v in snapshot233
               if not k.startswith("_") and
               re.search(r"beta|causal|target|impulse|shrink|baseline|frontier|^fits|^Bdir|^Binput", k, re.I)]
for key233, value233 in selected233[:100]:
    if isinstance(value233, (np.ndarray, torch.Tensor)):
        print(key233, type(value233).__name__, tuple(value233.shape))
    elif isinstance(value233, (float, int, str, dict, list, tuple)):
        print(key233, type(value233).__name__, str(value233)[:240])

print("\nCURRENT EDITS", edited())
print("b_in shape:", tuple(load_original("core.b_in.weight").shape))
with Path("/work/notes.md").open("a") as file233:
    file233.write(
        "\n## Run 233: baseline fails fresh factual validation\n"
        "Forward27 baseline77 NLL=1.63635 versus original=1.63043, cost .00592 "
        "on the same fresh 100 tuning stays; baseline itself lacks factual safety. "
        "Recovering architecture, record-derived targets, and prior calibration frontier "
        "before choosing a smaller or more selective edit. No weights changed. "
        "Python runs: 233; forwards: 27.\n"
    )
```