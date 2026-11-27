The contraction improved doubled-insulin prediction from about −19.55 to −15.75, but half-dose insulin remains far too weak and doubled norepinephrine worsened. Scale is therefore only part of the problem. I’ll preserve this candidate and test the retained fit that more strongly constrains total dose responses. Improvement across both directions, within the factual limit, would justify keeping it.

```python
import numpy as np, torch, hashlib
from pathlib import Path

checkpoint285 = {n: torch.as_tensor(load(n), dtype=torch.float32).clone()
                 for n in edited()}
results285 = {
    "contracted_0.02": {
        "nll_delta": 0.00175,
        "shifts": [-15.750, 1.310, 0.744, -1.195],
    },
    "checkpoint273": {
        "nll_delta": 0.00096,
        "shifts": [-19.551, 1.037, 1.719, -1.206],
    },
}
gamma285 = 10.0
candidate285 = candidates267[gamma285]
for n, v in checkpoint273.items():
    save_edit(n, torch.as_tensor(v, dtype=torch.float32))
save_edit("core.prelude.0.ffn.2.weight",
          torch.as_tensor(candidate285["weight"], dtype=torch.float32))
save_edit("core.prelude.0.ffn.2.bias",
          torch.as_tensor(candidate285["bias"], dtype=torch.float32))

print("gamma", gamma285,
      "held-out normalized-state RMS", candidate285["E_rmse"],
      "contrast relative error", candidate285["contrast_relative"])
print("installed FFN shape",
      tuple(torch.as_tensor(load("core.prelude.0.ffn.2.weight")).shape))
hash285 = hashlib.sha256(b"".join(
    np.asarray(load(n), dtype=np.float32).tobytes() for n in sorted(edited())
)).hexdigest()[:16]
print("EDIT HASH", hash285)

with open("/work/notes.md", "a") as f:
    f.write("\nForwards 25-26: output scale .02 factual delta +.00175; dose shifts (-15.750,+1.310,+.744,-1.195), versus record references (-12.9555,+6.4778,+4.2797,-2.1398). Scale improves main readout but does not solve asymmetry.\n")
    f.write(f"Python run 285; forwards 26. Preserved contracted candidate in checkpoint285. Installed retained total-response FFN candidate gamma={gamma285}, with all other weights restored from checkpoint273. Stronger total-response constraint tested next with factual safeguard. Hash {hash285}.\n")
```
