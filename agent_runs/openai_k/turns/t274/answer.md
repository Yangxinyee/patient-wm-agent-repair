The last test shows that the recurrent edit strengthens drug responses, but also perturbs factual predictions and produces an asymmetric norepinephrine response. I’ll recover the record-based effect estimates and candidate helpers before choosing an edit strength. The hypothesis is supported only if an official rollout improves toward those estimates while factual NLL increases by no more than 0.005 nats; otherwise this candidate needs revision.

```python
from pathlib import Path
import inspect, re, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

r246_notes = Path("/work/notes.md").read_text()
r246_lines = r246_notes.splitlines()
print("\n=== EARLIER IDENTIFICATION AND EFFECT EVIDENCE ===")
r246_matches = [
    (i, line) for i, line in enumerate(r246_lines)
    if re.search(r"truth|true effect|identif|randomiz|instrument|one.hour|target|effect estimate|causal estimate|confound", line, re.I)
]
for i, line in r246_matches[:65]:
    print(f"{i}: " + " ".join(r246_lines[max(0, i-1):min(len(r246_lines), i+3)])[:650])

print("\n=== CURRENT EDITS ===")
print(edited())
print("\n=== RECENT KERNEL OBJECTS ===")
r246_objects = list(globals().items())
for r246_name, r246_obj in r246_objects[-110:]:
    if r246_name.startswith(("r246_", "__")):
        continue
    if inspect.isfunction(r246_obj):
        r246_code = r246_obj.__code__
        print(r246_name, "function", r246_code.co_varnames[:r246_code.co_argcount])
    elif hasattr(r246_obj, "shape"):
        print(r246_name, type(r246_obj).__name__, tuple(r246_obj.shape))
    elif isinstance(r246_obj, dict):
        print(r246_name, "dict", list(r246_obj.keys())[:18])
    elif isinstance(r246_obj, (list, tuple)):
        print(r246_name, type(r246_obj).__name__, "length", len(r246_obj))
    elif isinstance(r246_obj, (int, float, str)):
        print(r246_name, repr(r246_obj)[:220])
    else:
        print(r246_name, type(r246_obj).__name__)

with open("/work/notes.md", "a") as r246_file:
    r246_file.write(
        "\n## Run 246: restart recovery before recurrent validation\n"
        "- Run 245 shows controllable local dose response, but factual perturbations and NE asymmetry remain unresolved.\n"
        "- Recovering architecture, record-identification evidence, and candidate interfaces; no weights changed.\n"
        "- Python runs: 246; forward passes: 17.\n"
    )
```
