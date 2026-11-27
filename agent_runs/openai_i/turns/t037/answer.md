The architecture confirms that doses reach the recurrent state through the action encoder, latent posterior, and prelude. Scaling the encoder may therefore be normalized away rather than strengthen drug effects. I’ll recover the dose-response estimates and probe stay IDs next; a paired rollout will distinguish useful amplification from an ineffective scaling change.

```python
from pathlib import Path
import numpy as np
import pandas as pd

_notes = Path("/work/notes.md").read_text()
print("=== EARLY CAUSAL ANALYSIS, CONTINUED ===")
print(_notes[4300:8000])

print("\n=== RETAINED DATA AND HELPERS ===")
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_"):
        continue
    if isinstance(_obj, (pd.DataFrame, np.ndarray)):
        print(_name, type(_obj).__name__, _obj.shape)
    elif isinstance(_obj, dict):
        print(_name, "dict", list(_obj.keys())[:6])
    elif callable(_obj) and getattr(_obj, "__module__", "") == "__main__":
        print(_name, "function")
    elif isinstance(_obj, (list, tuple)) and (
        "stay" in _name.lower() or "id" in _name.lower()
    ):
        print(_name, repr(_obj)[:800])

_restart_tuning_ids = pd.read_csv(
    "/data/tuning.csv", usecols=["stay_id"]
)["stay_id"].drop_duplicates().tolist()
print("\nFirst 100 tuning stay IDs:", _restart_tuning_ids[:100])

print("\n=== ACTION / POSTERIOR / PRELUDE TENSORS ===")
for _name in NAMES:
    if any(_s in _name for _s in (
        "encode_action", "z_post", "z_in", "prelude", "z_head", "attr_heads.value"
    )):
        print(_name, SHAPES[_name])

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Python 31; forward 5\n"
        "- Architecture recovered: action effects can enter through z_post and prelude.\n"
        "- Recovering causal-analysis details, retained helpers, and tuning stay IDs "
        "after the previous display was truncated.\n"
    )
```