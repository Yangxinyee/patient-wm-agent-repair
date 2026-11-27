The architecture confirms that drug information reaches the recurrent state through both the action embedding and the latent update. A shared sign reversal therefore need not repair both drugs. I’ll inspect the dose-channel construction and retained candidates separately; a useful candidate must improve official rollout effects while staying within the factual-loss limit.

```python
import re, inspect
from pathlib import Path
import numpy as np
import torch

g208 = dict(globals())
print("=== RECENT FUNCTIONS ===")
recent208 = []
for name, obj in g208.items():
    nums = re.findall(r"\d+", name)
    if inspect.isfunction(obj) and nums and max(map(int, nums)) >= 130:
        recent208.append(name)
print(recent208)

print("\n=== DOSE CHANNEL CONSTRUCTION ===")
for name in ("r132_channels", "r131_f", "lin0", "mlp0", "norm0"):
    obj = g208.get(name)
    if inspect.isfunction(obj):
        try:
            text = inspect.getsource(obj)
        except Exception:
            text = repr(obj.__code__.co_names)
        print(name, text[:2300])

print("\n=== DATAFRAMES ===")
for name, obj in g208.items():
    if hasattr(obj, "columns") and hasattr(obj, "shape"):
        print(name, obj.shape, list(obj.columns)[:24])

print("\n=== RECENT CANDIDATE OBJECTS ===")
for name in ("baseline138", "candidate139", "saved139", "candidate193",
             "backup199", "best188", "r132_targets", "contexts_audit"):
    obj = g208.get(name)
    if isinstance(obj, dict):
        print(name, "keys", list(obj)[:20])
    elif torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        print(name, tuple(obj.shape))
    else:
        print(name, type(obj).__name__)
print("\n=== RELATED OBJECT NAMES ===")
print([name for name in g208 if any(s in name.lower() for s in
       ("132", "138", "139", "188", "205", "206", "action_audit"))][:100])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 208: focused dose-channel recovery\n"
            "- Architecture confirms parallel direct-action and posterior-latent routes; a common latent sign reversal is not a drug-specific causal calibration.\n"
            "- Recovering exact physical-channel construction and candidate interfaces with bounded output.\n"
            "- No weights changed. Python runs: 208; forward passes: 14.\n")
```