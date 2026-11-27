The second factual batch increased NLL by **0.00380**, also within tolerance. That supports retaining this candidate as a fallback, but norepinephrine doubling still falls short. I’ll inspect the existing edit machinery for a more selective correction; it must improve doubling without sacrificing factual fit or the already-good half-dose response.

```python
import inspect
import dis
import types
from pathlib import Path
import numpy as np

with open("/work/notes.md", "a") as audit228_log:
    audit228_log.write(
        "\n## Forward 23 and independent factual validation\n"
        "- Gain-8 rollout, first 100 tuning: insulin x2 -11.079, half +5.813; "
        "NE x2 +1.019, half -1.695.\n"
        "- Forward 24 (directory fwd_25; intervening invalid request uncounted): "
        "next 100 tuning factual value dNLL +0.00380.\n"
        "- Both factual batches pass, but NE doubling remains substantially below "
        "the record-based kernel estimate. Candidate retained as fallback.\n"
    )

print("HEAD BUILDER")
print("signature:", inspect.signature(head177_make))
print("globals used:", head177_make.__code__.co_names)
try:
    print(inspect.getsource(head177_make))
except (OSError, TypeError):
    print(dis.Bytecode(head177_make).dis())

print("\nHEAD EVALUATOR")
print(inspect.signature(head177_eval), head177_eval.__code__.co_names)

print("\nAVAILABLE FIT/SIMULATION HELPERS")
audit228_helpers = [
    n for n, v in list(globals().items())
    if isinstance(v, types.FunctionType)
    and not n.startswith("__")
]
print(audit228_helpers)

print("\nRESERVED NEURONS:", head177_free)
print("\nRECENT MAP CONSTRUCTION NOTES")
audit228_notes = Path("/work/notes.md").read_text()
for audit228_marker in ("## Run 177", "## Run 178", "## Run 184", "## Run 185"):
    audit228_pos = audit228_notes.find(audit228_marker)
    if audit228_pos >= 0:
        audit228_end = audit228_notes.find("\n## ", audit228_pos + 5)
        print(audit228_notes[audit228_pos:
              min(audit228_end if audit228_end >= 0 else len(audit228_notes),
                  audit228_pos + 1000)])
```
