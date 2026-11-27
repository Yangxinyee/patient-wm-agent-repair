The retained action-trace fits are available, but reading-bundle reconstruction is still unresolved. One possible cause is how infusion END atoms are assigned to the two encoders. I’ll inspect the cache layout and event representation, then test that hypothesis against the exported posterior means.

```python
import numpy as np
import torch
from pathlib import Path

def r78_desc(x):
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return f"{type(x).__name__}{tuple(x.shape)}"
    if isinstance(x, dict):
        return "dict " + str({str(k)[:35]: r78_desc(v) for k, v in list(x.items())[:12]})
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}] first={str(x[:1])[:120]}"
    return str(x)[:100]

print("=== CACHE ===")
print(r78_desc(globals().get("cache")))
print("=== RECONSTRUCTION OBJECTS ===")
for _prefix in ("r33", "r35", "r36"):
    for _n in sorted(n for n in list(globals()) if n.startswith(_prefix)):
        _v = globals()[_n]
        if not callable(_v):
            print(_n, r78_desc(_v))
print("=== HELPER CONSTANTS/GLOBAL REFERENCES ===")
for _n in ("r36_atom", "r36_set_encode", "rr50_action_parts",
           "rr58_observation_encoding", "r72_posterior", "r33_action_embedding"):
    _fn = globals().get(_n)
    if callable(_fn):
        print(_n, "globals", _fn.__code__.co_names, "constants",
              [str(x)[:100] for x in _fn.__code__.co_consts])
print("=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text()[:2500])
print("=== VOCAB REPRESENTATION ===")
for _key in ("codes", "event_kinds", "atom_kinds", "numeric_note"):
    print(_key, str(VOCAB[_key])[:1200])
print("numeric keys", list(VOCAB["numeric"]))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 78\nAction traces recovered. Testing whether END-atom encoder assignment explains reading-only posterior mismatch; no edit. Counts: 78 Python runs, 6 forward passes.\n")
```