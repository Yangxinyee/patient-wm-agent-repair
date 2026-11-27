The recovered notes estimate an immediate glucose decrease of about 1.9–2.0 per insulin unit and an arterial MAP increase of about 3 per norepinephrine rate unit. These are provisional record-based estimates, not the full rollout targets. The output was truncated before the candidate inventory, so I’ll retrieve that compactly; a usable candidate must reproduce the estimated dynamics, not just the immediate signs.

```python
import inspect, re, numpy as np

r247_lines = Path("/work/notes.md").read_text().splitlines()
print("=== EARLY DYNAMIC EVIDENCE ===")
print("\n".join(r247_lines[28:46]))
print("\n=== ROLLOUT TARGET NOTES ===")
r247_selected = [
    s for s in r247_lines
    if re.search(r"impulse|deconvol|IRF|estimated.*rollout|rollout.*target|true.*shift|record-estimated", s, re.I)
]
print("\n".join(r247_selected)[:2400])

r247_items = list(globals().items())
print("\n=== RECENT OBJECT INVENTORY ===")
for r247_n, r247_o in r247_items[-105:]:
    if r247_n.startswith(("r246_", "r247_", "__")):
        continue
    if inspect.isfunction(r247_o):
        print(r247_n, "fn", r247_o.__code__.co_varnames[:r247_o.__code__.co_argcount])
    elif hasattr(r247_o, "shape"):
        print(r247_n, type(r247_o).__name__, tuple(r247_o.shape))
    elif isinstance(r247_o, dict):
        print(r247_n, "dict", list(r247_o)[:9])
    elif isinstance(r247_o, (tuple, list)):
        print(r247_n, type(r247_o).__name__, len(r247_o))
    elif isinstance(r247_o, (float, int, str)):
        print(r247_n, repr(r247_o)[:100])

print("\n=== REPLAY FUNCTIONS ===")
r247_fns = [
    (n, o) for n, o in r247_items
    if inspect.isfunction(o) and re.search(r"replay|core|head|coda|read|mean|prelude", n, re.I)
]
for r247_n, r247_o in r247_fns[-20:]:
    print(r247_n, r247_o.__code__.co_varnames[:r247_o.__code__.co_argcount])

with open("/work/notes.md", "a") as r247_file:
    r247_file.write(
        "\n## Run 247: compact candidate and dynamic-target recovery\n"
        "- Immediate cross-fitted effects recovered, but they are not full-rollout targets.\n"
        "- Run 246 output was truncated; recovering compact helper and candidate inventory.\n"
        "- No weights changed. Python runs: 247; forward passes: 17.\n"
    )
```
