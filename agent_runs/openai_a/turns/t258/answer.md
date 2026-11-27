Scale 6 is restored, and the recovered helpers show how the dose contrast is inserted into the value head. This does not yet establish a better edit. I’ll inspect the fitted contrast and cache provenance before screening a small intercept correction. Consistent factual gains across separate caches would support testing it on the GPU; disagreement would reject it.

```python
import numpy as np
from pathlib import Path

def repair223_summary(v, depth=0):
    if hasattr(v, "shape"):
        return {"type": type(v).__name__, "shape": tuple(v.shape)}
    if isinstance(v, dict):
        if depth >= 2:
            return {"keys": list(v)[:8], "length": len(v)}
        return {str(k): repair223_summary(x, depth + 1)
                for k, x in list(v.items())[:8]}
    if isinstance(v, (list, tuple)):
        return {"length": len(v),
                "first": [repair223_summary(x, depth + 1) for x in v[:2]]}
    return str(v)[:180]

for _n in ("r3_candidates", "r3_choice", "r3_applied",
           "sc_nodes", "sc_rank", "sc_bias_comp",
           "selective_trial_lam", "selective_trial_scale"):
    print(_n, repair223_summary(globals()[_n]))

print("DYNAMIC INSULIN PARAMETERS:", rd_dynamic_fit.x)

for _label in ("first", "extra", "tuning"):
    _c = sc_fact[_label]
    print("\nCACHE", _label)
    print({k: repair223_summary(_c[k]) for k in
           ("x", "logits", "gate", "bins", "original_nll", "meta")})
    if hasattr(_c["meta"], "columns"):
        print("metadata columns:", _c["meta"].columns.tolist())
        _idc = next((c for c in ("stay_id", "stay") if c in _c["meta"].columns), None)
        if _idc:
            print("stay range/count:", _c["meta"][_idc].min(),
                  _c["meta"][_idc].max(), _c["meta"][_idc].nunique())

_notes223 = Path("/work/notes.md").read_text()
_sections223 = _notes223.split("\n## ")
_evidence223 = [
    s for s in _sections223
    if any(term in s.split("\n", 1)[0].lower()
           for term in ("regression discontinuity", "fuzzy", "flexible ne", "ne identification"))
]
print("\nTARGETED CAUSAL NOTEBOOK SECTIONS")
for _s in _evidence223[:4]:
    print("##", _s[:650])
```
