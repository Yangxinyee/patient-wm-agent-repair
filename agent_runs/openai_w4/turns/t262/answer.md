The record-derived targets are approximately −7.8 and +3.9 for insulin, and +2.2 and −1.1 for norepinephrine on that tuning subset. The current edit’s insulin-halving response is much too large, so simply increasing strength is unlikely to help. I’ll inspect how the dose-innovation channels are centered and injected; altered feedback through that centering is a testable explanation.

```python
import inspect, numpy as np
from pathlib import Path

print("=== CHANNEL AND FACTUAL EVALUATION IMPLEMENTATIONS ===")
budget229 = 6500
for name229 in ["collect_channel186", "factual_eval194", "r216_fn", "gated_ne185",
                "f200", "cpu_factual129"]:
    obj229 = globals().get(name229)
    if obj229 is None:
        continue
    try:
        source229 = inspect.getsource(obj229)
    except Exception:
        source229 = ("signature=" + str(inspect.signature(obj229)) +
                     "\n referenced names=" + repr(obj229.__code__.co_names) +
                     "\n constants=" + repr(obj229.__code__.co_consts)[:500])
    excerpt229 = source229[:min(2200, budget229)]
    print(name229, "\n", excerpt229)
    budget229 -= len(excerpt229)
    if budget229 <= 0:
        break

print("\n=== CHANNEL CONSTRUCTION OBJECTS ===")
for name229 in ["dual_channel186", "target_ins186", "target_ne206", "candidate193",
                "dual_candidate193", "candidates225", "backup_before210"]:
    if name229 not in globals():
        continue
    obj229 = globals()[name229]
    if isinstance(obj229, dict):
        print(name229, "keys=", list(obj229)[:18])
    elif hasattr(obj229, "shape"):
        print(name229, tuple(obj229.shape))
    else:
        print(name229, repr(obj229)[:180])

print("\n=== RECENT CONSTRUCTION NOTES ===")
for section229 in re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M):
    if re.match(r"## Run (18[5-9]|19[0-9]|200)\b", section229):
        print(section229[:320])

with open("/work/notes.md", "a") as f229:
    f229.write("\n## Run 229\nRecovered record-derived tuning targets: insulin x2 about -7.8, half +3.9; NE x2 about +2.2, half -1.1 (subset/equal-stay aggregation). Current insulin half response is excessive. Inspecting innovation centering and response injection to test feedback distortion rather than blindly increasing gains. No saved weights changed.\n")
```