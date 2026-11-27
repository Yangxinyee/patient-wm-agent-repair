The architecture confirms that the edit changes recurrent state, while the value decoder also uses prior-reading registers. Earlier record analyses support a negative insulin effect and a positive norepinephrine effect—not their raw associations. I’ll recover the calibration inputs and decoder helpers next. A local decoder must reproduce the GPU results before I trust any calibration.

```python
import inspect, re, numpy as np
from pathlib import Path

print("LIVE FUNCTIONS")
funcs164 = []
for k, v in list(globals().items()):
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            funcs164.append(k + str(inspect.signature(v)))
        except Exception:
            funcs164.append(k)
print("\n".join(funcs164)[:3500])

print("\nRECENT CALIBRATION OBJECTS")
for k, v in list(globals().items()):
    if re.search(r"(159|160|161|162)$", k):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", len(v), list(v)[:8])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v), repr(v[:3])[:120])
        elif not callable(v):
            print(k, type(v).__name__, repr(v)[:130])

print("\nDECODER TENSORS")
for n in NAMES:
    if any(t in n for t in [
        "attr_heads.value", "last_numeric", "last_status",
        "embed_transition.value", "query_norm", "z_head", "type_emb"
    ]):
        print(n, SHAPES[n])

print("\nTARGETED NOTE SECTIONS")
sections164 = re.split(r"\n(?=## )", Path("/work/notes.md").read_text())
budget164 = 3300
for s in sections164:
    heading = s.splitlines()[0] if s else ""
    if re.search(r"(Run 77\b|Run 159\b|Run 160\b|Run 161\b|Run 162\b)", heading):
        out = s[:min(1100, budget164)]
        print(out)
        budget164 -= len(out)
        if budget164 <= 0:
            break

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 164: compact decoder/calibration recovery\n"
        "Recovered recent calibration objects, decoder tensor names, and helper signatures "
        "with bounded output after run163 truncation. No weights changed. "
        "Next gate is local/GPU decoder agreement before calibration. "
        "Python runs: 164; forwards: 22.\n"
    )
```