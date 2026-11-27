The last result shows that a large latent change can have almost no effect on predictions; the tensor scales alone do not explain why. I’ll recover the exact state equations and earlier numeric-dose experiments before choosing another edit. A useful route must transmit dose changes to predictions while preserving factual fit—not merely change hidden states.

```python
from pathlib import Path
import re, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes203 = Path("/work/notes.md").read_text()
sections203 = re.split(r"(?=^## )", notes203, flags=re.M)
print("\n=== EARLIER IDENTIFICATION AND NUMERIC-DOSE RESULTS ===")
selected203 = []
for sec in sections203:
    heading = sec.splitlines()[0] if sec.splitlines() else ""
    if re.search(r"numeric|5x|random|identif|causal|kernel|confound", sec, re.I):
        selected203.append(sec)
early203 = selected203[:5]
numeric203 = [s for s in selected203
              if re.search(r"numeric|5x", s, re.I)
              and not re.search(r"Run (17[3-9]|18\d|19\d|20\d)\b", s.splitlines()[0])]
text203 = "\n".join(early203 + numeric203[-8:])
print(text203[:12500])

print("\n=== INPUT / CORE / READOUT TENSOR SHAPES ===")
for name in NAMES:
    if re.search(r"num|dose|rate|action|core\.(delta|a_in|b_in|c_out|prelude|z_in)|value", name, re.I):
        print(name, SHAPES[name])

print("\n=== RELEVANT RETAINED HELPERS AND CACHES ===")
for name in sorted(list(globals())):
    if name.startswith(("rx_", "carrier", "rp_", "rr_")):
        obj = globals()[name]
        if callable(obj):
            print(name, "function", getattr(getattr(obj, "__code__", None), "co_varnames", ())[:12])
        elif isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:8])
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, tuple(obj.shape))
        elif isinstance(obj, (str, int, float, bool)):
            print(name, repr(obj)[:140])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 203\nRecovered architecture and prior numeric-dose/identification summaries to choose a direct, testable transmission pathway. All weights remain original. Counts: 203 Python runs, 18 forwards.\n")
```