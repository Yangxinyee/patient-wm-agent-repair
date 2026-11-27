The saved nonlinear detector is designed to leave initial admissions unchanged, and a MAP response direction is available. The current joint edit passes the latest factual test narrowly, so any addition must be small. I’ll check the detector’s dose units and response size before selecting that trial; the key question is whether it separates dose changes from factual variation.

```python
import re
import numpy as np
from pathlib import Path

_notes = Path("/work/notes.md").read_text()
_sections = re.split(r"(?=^## )", _notes, flags=re.M)
print("=== RECORD-DERIVED EFFECT ESTIMATES ===")
for _sec in _sections:
    if any(re.search(r"## Run " + str(n) + r"\b", _sec)
           for n in [69, 73, 75, 76, 77]):
        print(_sec[:1050])
print("=== PRELUDE OUTPUT TENSORS ===")
print([(n, SHAPES[n]) for n in NAMES if "prelude.0" in n])
with np.load("/work/ne_nonlinear250.npz") as _a:
    coef261 = _a["raw_coefficient"].copy()
with np.load("/work/prelude_direction255.npz") as _a:
    direction261 = _a["direction"].copy()
print("=== DETECTOR UNITS / CONTRASTS ===")
print("keys", list(D244), "dose2_244", dose2_244)
print("factual prediction RMS", float(np.sqrt(np.mean((X244 @ coef261)**2))))
for _k in D244:
    _x = np.asarray(D244[_k])
    _y = np.asarray(Y244[_k])
    _p = _x @ coef261 if _x.shape[1] == len(coef261) else None
    print(_k, "design", _x.shape,
          "target mean/range", float(np.mean(_y)), [float(np.min(_y)), float(np.max(_y))],
          "prediction mean/rms", None if _p is None else
          [float(np.mean(_p)), float(np.sqrt(np.mean(_p**2)))])
print("=== VALIDATION STAY SETS ===")
for _name in ["validation_ids", "VALID_IDS", "fresh_ids232", "ids219", "train_ids190"]:
    _v = globals().get(_name)
    if _v is not None:
        print(_name, len(_v), list(_v)[:5])
print("currently edited", edited())
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 261: calibrate nonlinear trial units\n"
             "Saved nonlinear detector and normalized MAP direction recovered. "
             "Checking actual dose-contrast units and prelude output tensor names before a small trial. "
             "No weights changed. Python runs: 261; forwards: 29.\n")
```