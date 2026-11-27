The second batch confirms a stronger insulin response, while norepinephrine remains weak. These predictions alone do not establish correctness. I’ll recover the training-record estimates and saved edit details; agreement across batches would support keeping the insulin edit, while disagreement would call its calibration into question.

```python
from pathlib import Path
import numpy as np, json

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("=== EARLY RECORD EVIDENCE ===")
print(_notes[:15500])
print("=== RELEVANT NOTE HEADINGS ===")
for _line in _notes.splitlines():
    if _line.startswith("##"):
        print(_line)
print("=== SAVED CANDIDATES ===")
for _fn in ["margin_validated400.npz", "validated_current400.npz",
            "map_regularized_trial.npz", "glucose_115_candidate.npz",
            "pre_recurrent_candidate.npz", "recurrent_glucose_valid.npz"]:
    with np.load("/work/" + _fn, allow_pickle=False) as _a:
        print(_fn, {k: (_a[k].shape, str(_a[k].dtype)) for k in _a.files})
print("=== KERNEL INTERFACES ===")
for _k in sorted(list(globals())):
    if not _k.startswith("_") and any(s in _k.lower() for s in
        ["target", "batch", "stay", "route", "ridge", "margin", "map_", "gluc", "fisher"]):
        _v = globals()[_k]
        if isinstance(_v, np.ndarray):
            print(_k, "array", _v.shape, _v.dtype)
        elif callable(_v):
            print(_k, "callable")
        elif isinstance(_v, (list, tuple, dict)) and len(_v) < 20:
            print(_k, repr(_v)[:1400])
        else:
            print(_k, type(_v).__name__, str(_v)[:300])
print("CURRENT EDITS:", edited())
with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Run 292: restart recovery\n"
             "- Forward 42 (folder fwd_43) confirms insulin response on a second batch "
             "(-9.287/+6.039), while NE remains weak (+0.508/-0.118).\n"
             "- Recovering architecture, training-record evidence, and exact candidate components. "
             "No weights changed.\n"
             "- Python runs: 292; forward passes: 42.\n")
```