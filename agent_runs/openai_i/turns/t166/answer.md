The retained estimates support negative insulin effects and positive norepinephrine effects after accounting for dose selection, but those are one-hour estimates. They do not establish the accumulated rollout targets. I’ll recover the lagged estimates next; persistent effects would require a recurrent repair rather than a next-reading adjustment.

```python
from pathlib import Path
import re
import json
import numpy as np

text149 = Path("/work/notes.md").read_text()
sections149 = re.split(r"(?m)^## ", text149)

print("=== EARLY SECTION TITLES ===")
for sec in sections149[1:36]:
    print(sec.splitlines()[0])

print("\n=== LAGGED EFFECT AND ROLLOUT EVIDENCE ===")
budget149 = 5900
for sec in sections149:
    title = sec.splitlines()[0] if sec.splitlines() else ""
    match = re.search(r"Python\s+(\d+)", title)
    if match and int(match.group(1)) > 105:
        continue
    if any(term in sec.lower() for term in
           ["cumulative", "distributed lag", "multi-hour", "impulse response",
            "impulse estimate", "rollout target", "target shift"]):
        excerpt = sec[:min(1700, budget149)]
        print("## " + excerpt)
        budget149 -= len(excerpt)
        if budget149 < 200:
            break

print("\n=== FORWARD FILES ===")
for p in sorted(Path("/work").glob("fwd_*")):
    print(p.name, [q.name for q in p.iterdir()])

print("\n=== VOCABULARY CONFIG ===")
print(VOCAB["config"])
print("numeric example:", list(numeric.items())[:1])
print("state IDs:", np.unique(sid)[:8], "hours:", hours[:12])

print("\n=== RECURRENT HELPER CONTRACT ===")
for fn in [core_step136, block136, attention136, mlp136, atom145, pool137]:
    print(fn.__name__, "args:", fn.__code__.co_varnames[:fn.__code__.co_argcount],
          "names:", fn.__code__.co_names,
          "constants:", [x for x in fn.__code__.co_consts
                         if isinstance(x, (str, int, float, bool))][:35])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python 149 — recover accumulated effects\n"
        "- Preliminary one-hour effects are not substituted for rollout targets.\n"
        "- Recovering lagged record evidence and helper contracts; no edits active.\n"
    )
```
