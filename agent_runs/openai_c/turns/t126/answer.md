The latent-head edit passes the probe’s factual limit, but uses most of the allowance without improving the dose responses. I’ll remove it and inspect the saved state directions for a recurrent-path edit. That approach would be supported by stronger rollout effects with little factual change.

```python
from pathlib import Path
import numpy as np

while len(ops()) > 1:
    undo()
print("RESTORED:", ops())

print("\nACTION AND RECURRENT TENSORS")
for n in NAMES:
    if any(s in n for s in ["encode_action", "core.prelude", "core.b_in",
                             "core.c_out", "core.coda", "core.out_norm",
                             "z_post", "z_prior", "z_in"]):
        print(n, SHAPES[n])

print("\nSAVED STATE / EDIT OBJECTS")
rows = []
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if isinstance(v, np.ndarray):
        rows.append(f"{k}:{v.shape}")
    elif isinstance(v, dict) and k not in {"VOCAB", "SHAPES"}:
        rows.append(f"{k}:dict[{', '.join(map(str, list(v)[:6]))}]")
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        rows.append(f"{k}:function")
print("\n".join(rows)[-4500:])

print("\nEARLY NOTEBOOK")
print(Path("/work/notes.md").read_text()[:3800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 89: latent-head edit rejected\n"
            "- Forward 33 factual increase +0.00428; forward 34 showed no useful causal improvement.\n"
            "- Removed both z_head input gains; restored numeric gain 4 alone.\n"
            "- Investigating action/recurrent pathway amplification using saved conditional state directions.\n"
            "- Forward passes used: 34; 16 remain.\n")
```