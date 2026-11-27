Most stronger symmetric candidates exceeded the factual budget, but gain 1.5/gate 26 passed at +0.00370 CPU NLL. I’ll provisionally save that candidate and test its rollout. It should improve halving if one-sided gating was the cause; a much weaker doubling response would refute its usefulness.

```python
from pathlib import Path
import numpy as np

backup266 = {n:clone258(load(n)) for n in edited()}
choice266 = (1.5,26.0)
candidate266 = screen265[choice266]["weights"]
other_diffs266 = [n for n in NAMES
                  if n not in ("z_post.0.weight","z_post.0.bias")
                  and not np.array_equal(a262(load(n)),a262(candidate266[n]))]
assert not other_diffs266, other_diffs266
for n in ("z_post.0.weight","z_post.0.bias"):
    save_edit(n,a262(candidate266[n]).astype(np.float32))
print("SAVED SYMMETRIC INSULIN",choice266,
      {k:screen265[choice266][k] for k in ("nll","delta")})

nt266 = Path("/work/notes.md").read_text()
ss266 = nt266.find("## Run 212")
if ss266 < 0:
    ss266 = nt266.find("## Runs 210")
print("\nLATER REPAIR NOTES")
print(nt266[ss266:ss266+4400] if ss266 >= 0 else "section not found")
with open("/work/notes.md","a") as f:
    f.write("\n## Run 266\nProvisionally saved symmetric insulin pair, gain "
            "1.5/gate 26, changing only z_post.0.weight/bias. CPU dNLL "
            "+.00369549. Prior gain-1.5/gate-28 one-sided candidate backed "
            "up in backup266. GPU rollout and factual validation pending.\n")
```
