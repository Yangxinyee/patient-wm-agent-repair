The flow edit worsened factual NLL by 0.105 nats, so it is rejected. I’ll remove it and recover the earlier causal evidence and action-encoder details. The next correction must distinguish dose changes from ordinary patient-state variation; a broad memory change clearly does not.

```python
from pathlib import Path
import types

_current = ops()
if _current and _current[-1]["name"] == "core.flow.a":
    undo()
print("Restored operations:", ops())

print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:6500])

print("\n=== SELECTED TENSOR SHAPES ===")
for _n in NAMES:
    if any(_term in _n for _term in (
        "encode_action", "attr_heads.value", "last_numeric",
        "query_norm", "embed.numeric", "z_post", "z_prior",
        "z_head", "z_in", "out_norm"
    )):
        print(_n, SHAPES[_n])

print("\n=== SAVED FUNCTIONS ===")
print([_n for _n, _v in list(globals().items())
       if not _n.startswith("_") and isinstance(_v, types.FunctionType)])

print("\n=== SAVED DICTIONARIES ===")
for _n, _v in list(globals().items()):
    if not _n.startswith("_") and isinstance(_v, dict) and _n not in ("SHAPES", "VOCAB"):
        print(_n, "keys:", list(_v)[:12])

with open("/work/notes.md", "a") as _notes:
    _notes.write(
        "\n## Forward 25 / Python run 63: flow edit rejected\n"
        "- core.flow.a x1.15 factual NLL 1.73963 vs 1.63461, increase +0.10502.\n"
        "- Removed flow scaling; restored numeric gain 4 alone.\n"
        "- Recovering early causal evidence and selective action/readout designs.\n"
        "- Forward passes used: 25.\n"
    )
```