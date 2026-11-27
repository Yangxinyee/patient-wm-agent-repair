The cache contains original weights, so stale weights are ruled out. I’ll inspect the atom helper’s categorical and numeric additions and recover the earlier exact reconstruction cases. If those cases reveal a reproducible convention, I can move on to dose calibration; otherwise the helper remains unvalidated.

```python
import dis
import re
import numpy as np
import torch

print("=== ATOM HELPER: COMPACT BYTECODE ===")
instructions80 = [
    f"{i.offset}:{i.opname} {i.argrepr[:100]}"
    for i in dis.get_instructions(complete_atom57)
]
print(" | ".join(instructions80))

print("\n=== CATEGORICAL TENSOR SHAPES ===")
for name80 in NAMES:
    if "embed.categorical" in name80:
        print(name80, SHAPES[name80])

print("\n=== ATOM METADATA ===")
for name80 in ("event_info57", "numeric_specs", "code_ids",
               "gap_prefix41", "latent_empty_e"):
    value80 = globals().get(name80)
    if isinstance(value80, dict):
        print(name80, repr(value80)[:1700])
    elif isinstance(value80, torch.Tensor):
        print(name80, tuple(value80.shape))
    else:
        print(name80, repr(value80)[:500])

print("\n=== RETAINED EARLIER RECONSTRUCTION OBJECTS ===")
for name80, value80 in sorted(list(globals().items())):
    if not re.search(r"(5[3-9]|6[0-4])$", name80):
        continue
    if isinstance(value80, dict):
        print(name80, "dict", repr(list(value80.items())[:2])[:320])
    elif isinstance(value80, (list, tuple)):
        print(name80, "length", len(value80), "first", repr(value80[:1])[:250])
    elif isinstance(value80, (np.ndarray, torch.Tensor)):
        print(name80, tuple(value80.shape))
    elif isinstance(value80, str) and len(value80) > 150:
        print(name80, "source-string length", len(value80))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 80; forward 6 — atom-construction audit\n"
        "- Cached weights exactly matched originals, ruling out stale weight edits.\n"
        "- Inspecting inserted categorical/numeric fields and retained earlier "
        "exact reconstruction cases before using posterior features for editing.\n"
        "- No weights changed.\n"
    )
```