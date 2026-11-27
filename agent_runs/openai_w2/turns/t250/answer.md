The constrained edit improved the difficult-cohort NLL increase to 0.005093, but it remains slightly above the limit. I’ll keep the weights unchanged and recover the saved candidates and record-based targets. Acceptance requires a safer factual margin and rollout effects supported by the records.

```python
from pathlib import Path
import re
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA DESCRIPTION")
print(Path("/data/README.md").read_text())

notes214 = Path("/work/notes.md").read_text()
print("\nEARLIER IDENTIFICATION / TARGET NOTES")
paras214 = notes214.split("\n\n")
selected214 = [
    p for p in paras214
    if re.search(r"causal|threshold|instrument|persistence|target|impulse|random|confound|effect", p, re.I)
]
earlier214 = "\n\n".join(p for p in selected214 if p not in notes214[-10000:])
print(earlier214[:18000])

print("\nRELEVANT KERNEL OBJECTS")
items214 = list(globals().items())
matches214 = []
for name214, obj214 in items214:
    if name214.startswith("_"):
        continue
    if not re.search(r"target|effect|impulse|rho|beta|causal|validated|fallback|candidate|record|212|206|204|176|151", name214, re.I):
        continue
    if callable(obj214):
        continue
    shape214 = getattr(obj214, "shape", None)
    desc214 = f"{name214}: {type(obj214).__name__}"
    if shape214 is not None:
        desc214 += f" shape={tuple(shape214)}"
        if isinstance(obj214, np.ndarray) and obj214.size <= 32:
            desc214 += f" values={obj214!r}"
    elif isinstance(obj214, (float, int, str, bool, np.number)):
        desc214 += f" value={str(obj214)[:250]}"
    elif isinstance(obj214, dict):
        desc214 += f" keys={list(obj214.keys())[:24]}"
    elif isinstance(obj214, (list, tuple)):
        desc214 += f" len={len(obj214)}"
        if len(obj214) <= 12:
            desc214 += f" value={repr(obj214)[:600]}"
    matches214.append(desc214)
print("\n".join(matches214[:180]))
print("\nEDITED TENSORS", edited())

with open("/work/notes.md", "a") as f214:
    f214.write(
        "\n\n## Forward26 / runs213–214: constrained trial needs more margin\n"
        "Difficult-cohort factual NLL delta +0.0050932033, "
        "stay-cluster SE 0.0014217141. By-event deltas: finger +0.004080, "
        "lab +0.000902, arterial +0.004678, cuff +0.007451. "
        "Improves on candidate_B204 but remains above 0.005 on this cohort. "
        "No weight changes; recovering candidates and record-based targets.\n"
    )
```
