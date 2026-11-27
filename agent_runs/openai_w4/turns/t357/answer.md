Norepinephrine doubling now matches the estimated effect closely, but halving overshoots; insulin responses are too small. Factual fit has room to strengthen the insulin correction. I’ll restore its previous strength while retaining the reversed norepinephrine correction, and remove unchanged tensor copies. Larger insulin responses with NLL still below the limit would support this revision.

```python
import torch as _T298
from pathlib import Path as _P298
import re as _R298

_changed298 = []
for _n298 in edited():
    _a298 = _T298.as_tensor(load(_n298))
    _b298 = _T298.as_tensor(load_original(_n298))
    if not _T298.equal(_a298, _b298):
        _changed298.append(_n298)
print("ACTUAL_CHANGED_TENSORS", _changed298)
print("CHANGES_OUTSIDE_PRIOR_TEN", sorted(set(_changed298) - set(backup297)))

backup298 = {n: _T298.as_tensor(load(n)).clone() for n in _changed298}

def apply_sparse298(weights):
    prepared = {}
    for name, value in weights.items():
        value = _T298.as_tensor(value).to(dtype=_T298.float32).clone()
        assert name in NAMES
        assert tuple(value.shape) == tuple(SHAPES[name])
        assert bool(_T298.isfinite(value).all())
        if not _T298.equal(value, _T298.as_tensor(load_original(name))):
            prepared[name] = value
    for name in list(edited()):
        drop_edit(name)
    for name, value in prepared.items():
        save_edit(name, value)
    return sorted(prepared)

params298 = (3.0, 0.0, 27.0, -0.5)
candidate298 = asymmetric_ins279(*params298)
print("SAVED_CANDIDATE", params298)
print("SPARSE_EDITS", apply_sparse298(candidate298))

_notes298 = _P298("/work/notes.md").read_text()
_lines298 = [
    x for x in _notes298.splitlines()
    if _R298.search(r"matched.*target|kernel.*target|identification assumption|confound", x, _R298.I)
]
print("TARGET_EVIDENCE", "\n".join(x[:700] for x in _lines298[-5:]))

with _P298("/work/notes.md").open("a") as _f298:
    _f298.write(
        "\n## GPU passes 42–43; Run 298\n"
        "Candidate (2.75,0,28,-0.5) passed first-100 tuning factual fit: "
        "original 1.59524, edited 1.59802, penalty +0.00278. "
        "Its rollout shifts were insulin -4.859/+2.825, NE +5.016/-4.777. "
        "NE doubling is close to the record-derived target; NE halving overshoots, "
        "and insulin is too weak. Actual changed tensors audited; unchanged saved "
        "copies removed. Backed up as backup298. Saved provisional candidate "
        "(3,0,27,-0.5), strengthening insulin while retaining reversed NE. "
        "GPU factual and rollout validation pending; no new causal evidence.\n"
    )
```
