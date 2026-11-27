The recovered records support opposite drug effects after pre-dose adjustment, but those one-step associations do not yet calibrate the rollout. The output was truncated before the carrier details. I’ll retrieve those details and the later dynamic estimates compactly; I need both a record-based target and a factual-safe route for transmitting it.

```python
import re
from pathlib import Path

text237 = Path("/work/notes.md").read_text()
lines237 = text237.splitlines()
patterns237 = r"dynamic|impulse|persistence|structural|target|identif|long.run|cumulative"
hits237 = [
    (i, line) for i, line in enumerate(lines237)
    if re.search(patterns237, line, re.I)
]
print("DYNAMIC/EFFECT EVIDENCE")
budget237 = 3800
for i, line in hits237:
    heading = next(
        (lines237[j] for j in range(i, -1, -1) if lines237[j].startswith("## ")),
        ""
    )
    s = heading + "\n" + line[:650] + "\n"
    if len(s) > budget237:
        break
    print(s)
    budget237 -= len(s)

print("\nCURRENT CACHE STRUCTURE")
for name in ["carrier211", "relay207", "rp_X", "rp_input", "rr_input4",
             "train213", "hold213", "desc208"]:
    if name in globals():
        print(name, describe236(globals()[name]))
if "cf215" in globals():
    key237 = next(iter(cf215))
    print("cf215 example", key237, describe236(cf215[key237]))
if "marker215" in globals():
    print("marker215", describe236(marker215))

print("\nPOSTERIOR HELPERS")
if "rx_show_function" in globals():
    for name, limit in [("post208", 2400), ("posterior211", 800),
                        ("fixedpost215", 800)]:
        try:
            rx_show_function(name, limit)
        except Exception as exc:
            print(name, type(exc).__name__, str(exc)[:120])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## Run 237\n"
        "Run236 output was truncated before implementation details. "
        "Retrieving compact dynamic-effect evidence and safe posterior/cache "
        "definitions. Early adjusted associations support signs but are not "
        "being treated as multi-hour causal targets. No weights changed. "
        "Counts: 237 Python runs, 23 forwards.\n"
    )
```