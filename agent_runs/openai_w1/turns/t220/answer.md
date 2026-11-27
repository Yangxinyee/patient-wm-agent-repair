The stronger gates eliminate corrections on bundles without the drug, but we have not shown that the correction reaches future predictions. I’ll recover the exact state-update equations and the compensation fit before editing weights. The next candidate must transmit dose changes while preserving factual predictions.

```python
import numpy as np, torch, re
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECENT CANDIDATE VARIABLES ===")
keys197 = sorted(k for k in list(globals())
                 if re.match(r"^(r19[356]|rp_|saved_posterior|carrier|comp|obs_dir)", k))
for k in keys197:
    v = globals()[k]
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys())[:24])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "len", len(v), repr(v)[:220])
    elif isinstance(v, (str, int, float)):
        print(k, repr(v)[:300])
    else:
        print(k, type(v).__name__)

print("\n=== EARLIER IDENTIFICATION NOTES ===")
notes197 = Path("/work/notes.md").read_text()
sections197 = re.split(r"(?=^## )", notes197, flags=re.M)
picked197 = [s for s in sections197
             if re.search(r"causal|random|instrument|confound|kernel|identif|policy|overlap", s, re.I)]
print("\n".join(picked197[:18])[:10500])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 197\nStrong presence gates eliminate inactive-bundle leakage. Recovering exact state equations and latent compensation before installing a direct carrier; observation directions alone are not proof of causal transmission. Counts: 197 Python runs, 16 forwards.\n")
```