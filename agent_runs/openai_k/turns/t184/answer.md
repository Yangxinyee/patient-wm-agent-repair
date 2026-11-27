The architecture confirms that drug information passes through the posterior, recurrent state, and shared prediction heads. The previous output was truncated before the retained estimates appeared. I’ll retrieve those estimates and the editing helpers compactly; they should show whether a controlled dose channel is feasible without broad factual disruption.

```python
import re, inspect
from pathlib import Path

r168_items = list(globals().items())
print("=== RECENT HELPERS ===")
for k, v in r168_items:
    nums = re.findall(r"\d+", k)
    if inspect.isfunction(v) and nums and int(nums[-1]) >= 130:
        print(k, inspect.signature(v))

print("\n=== RECENT VARIABLES ===")
r168_names = []
for k, v in r168_items:
    nums = re.findall(r"\d+", k)
    if not k.startswith("_") and not inspect.isfunction(v) and not inspect.ismodule(v):
        if (nums and 130 <= int(nums[-1]) <= 166) or any(
            w in k.lower() for w in ["causal", "effect", "dynamic", "ident", "target", "panel"]
        ):
            r168_names.append(k)
print(" ".join(sorted(r168_names))[:3000])

print("\n=== RELEVANT TENSOR SHAPES ===")
for k in NAMES:
    if any(w in k for w in ["z_post", "z_in", "attr_heads.value_bin",
                            "query_norm", "last_numeric", "core.flow",
                            "core.delta_in", "core.a_in", "z_head"]):
        print(k, SHAPES[k])

print("\n=== RECORD-BASED IDENTIFICATION NOTES ===")
r168_lines = Path("/work/notes.md").read_text().splitlines()
r168_selected = []
r168_heading = ""
for line in r168_lines:
    if line.startswith("##"):
        r168_heading = line
    if any(w in line.lower() for w in
           ["causal", "instrument", "randomiz", "persistence", "rho",
            "per unit", "per rate", "long-run", "long run", "confound",
            "negative control", "target shift", "identified effect"]):
        r168_selected.append(r168_heading + "\n" + line[:500])
print("\n".join(r168_selected)[:3200])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 168: compact architecture and evidence recovery\n"
        "- Architecture confirms a nonlinear shared state/readout path; run 167 "
        "was truncated before retained variables and evidence.\n"
        "- Retrieving only recent interfaces, relevant tensor shapes, and "
        "record-based identification notes. No weights changed.\n"
        "- Python runs: 168; forward passes: 10.\n"
    )
```