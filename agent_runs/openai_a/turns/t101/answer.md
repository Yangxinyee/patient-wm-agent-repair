The strong graft alternates between a large negative glucose shift and a rebound toward zero. That points to a rollout-feedback problem, not merely insufficient strength. I’ll reconstruct local state updates so we can test dose sensitivity separately from glucose feedback. The reconstruction must first reproduce the original GPU states; otherwise it is not trustworthy enough to design an edit.

```python
from pathlib import Path
import numpy as np

print("=== DYNAMIC / THRESHOLD IDENTIFICATION ===")
rr_notes = Path("/work/notes.md").read_text()
print(rr_notes[23378:26900])

print("\n=== STATE-UPDATE TENSORS ===")
for rr_n in NAMES:
    if rr_n.startswith(("embed.", "encode.", "encode_action.", "core.")):
        print(rr_n, SHAPES[rr_n])

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB.keys()))
for rr_k in ("config", "codes", "kinds", "event_kinds", "categorical"):
    if rr_k in VOCAB:
        print(rr_k, repr(VOCAB[rr_k])[:1000])

print("\n=== STATE FILES AND CPU PARTS ===")
for rr_k, rr_s in state_files.items():
    if isinstance(rr_s, dict) or hasattr(rr_s, "files"):
        rr_keys = list(rr_s.keys()) if isinstance(rr_s, dict) else rr_s.files
        print("state_files", rr_k,
              [(k, np.asarray(rr_s[k]).shape) for k in rr_keys])
for rr_k, rr_p in robust_parts.items():
    print("robust_parts", rr_k,
          [(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
           for k, v in rr_p.items()] if isinstance(rr_p, dict)
          else type(rr_p).__name__)

print("\n=== CPU FUNCTION DEPENDENCIES ===")
for rr_n in ("cpu_head_parts", "reading_metadata", "repair_head_features"):
    rr_f = globals().get(rr_n)
    if hasattr(rr_f, "__code__"):
        print(rr_n, "globals:", rr_f.__code__.co_names,
              "locals:", rr_f.__code__.co_varnames)

with Path("/work/notes.md").open("a") as rr_f:
    rr_f.write(
        "\n## Feedback diagnosis; Python 87, forwards 11/50\n"
        "- Strong insulin graft has alternating-hour responses, consistent with "
        "a harmful history-feedback component in its decoder.\n"
        "- Recomputed reached-hour then stay averages differ from earlier summaries; "
        "future comparisons must use identical weighting.\n"
        "- Proposed test: validate a CPU local-step reconstruction against original "
        "GPU states, then fit a small decoder on synthetic dose/history contrasts. "
        "No new weight edits yet.\n"
    )
```