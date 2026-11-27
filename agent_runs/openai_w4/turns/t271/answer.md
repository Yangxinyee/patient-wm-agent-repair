The retained notes distinguish immediate effects from longitudinal effects and explicitly flag uncertainty in the latter. I won’t treat a cached target as truth without checking its construction. Next I’ll retrieve the identification evidence and channel details; consistent estimates across methods would support further editing, while disagreement would favor a more conservative repair.

```python
from pathlib import Path
import inspect, re

text236 = Path("/work/notes.md").read_text()
lines236 = text236.splitlines()

print("=== EARLY RECORD EVIDENCE ===")
print("\n".join(lines236[:85])[:3700])

print("=== LATER CHANNEL CONSTRUCTION NOTES ===")
for start, end in [(179, 194), (195, 210)]:
    chunks = re.split(r"(?m)(?=^## )", text236)
    chosen = []
    for chunk in chunks:
        head = chunk.splitlines()[0] if chunk.splitlines() else ""
        nums = re.findall(r"\d+", head)
        if nums and start <= int(nums[0]) <= end:
            chosen.append(chunk)
    print("".join(chosen)[:2400])

print("=== ALL RELEVANT NAMES, WITHOUT LARGE VALUES ===")
names236 = [k for k, v in dict(globals()).items()
            if re.search(r"kernel|target|connection|channel|gate|ridge|dual|coef|innovation", k, re.I)
            and not inspect.isfunction(v)]
print(", ".join(sorted(names236))[:2400])

print("=== LOCAL EVALUATION DEPENDENCIES ===")
for name in ["factual_eval194", "collect_channel186", "collect_selected187",
             "gated_ne185", "replay_factual125"]:
    obj = globals().get(name)
    if inspect.isfunction(obj):
        print(name, obj.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write("""
## Run 236
Retrieving identification evidence and precise retained repair dependencies.
No weights changed. Longitudinal targets remain evidence-based estimates,
not simulator truth; aggregation must match the reached-row readout.
""")
```
