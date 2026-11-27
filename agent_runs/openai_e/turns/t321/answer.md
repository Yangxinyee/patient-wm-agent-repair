On this cohort, the record-derived targets are about −14.15, +7.07, +4.61 and −2.30. The archived alternatives had worse factual costs. I’ll test 90% of the current correction, expecting lower NLL loss with somewhat smaller dose responses. It remains unaccepted until paired factual checks show adequate margin.

```python
from pathlib import Path
import numpy as np

assert set(edited()) == {"core.b_in.weight"}, edited()
with np.load("/work/joint268.npz", allow_pickle=False) as _a274:
    _source274 = _a274["delta"].astype(np.float32).copy()
    _id274 = _a274["insulin_delta"].astype(np.float32).copy()
    _nd274 = _a274["ne_delta"].astype(np.float32).copy()
    _is274 = float(_a274["insulin_scale"])
    _ib274 = float(_a274["insulin_blend"])
    _ns274 = float(_a274["ne_scale"])

_original274 = np.asarray(load_original("core.b_in.weight"), dtype=np.float32)
_current274 = np.asarray(load("core.b_in.weight"), dtype=np.float32)
print("Current/archive maximum discrepancy:",
      float(np.max(np.abs(_current274-(_original274+_source274)))))
assert np.allclose(_current274, _original274+_source274, atol=2e-5, rtol=1e-6)

_lambda274 = 0.90
_delta274 = (_lambda274*_source274).astype(np.float32)
save_edit("core.b_in.weight", (_original274+_delta274).astype(np.float32))
np.savez("/work/joint274.npz",
         delta=_delta274,
         insulin_delta=(_lambda274*_id274).astype(np.float32),
         ne_delta=(_lambda274*_nd274).astype(np.float32),
         insulin_scale=_lambda274*_is274,
         insulin_blend=_ib274,
         ne_scale=_lambda274*_ns274,
         source_scale=_lambda274)

print("Installed insulin scale/blend, NE scale:",
      _lambda274*_is274, _ib274, _lambda274*_ns274)
print("Correction norm:", float(np.linalg.norm(_delta274)))
print("Edited tensors:", edited())

with Path("/work/notes.md").open("a") as _log274:
    _log274.write(
        "\n## Run 274: conservative joint shrinkage trial\n"
        "Forward35 exact-cohort impulse targets: insulin2=-14.149, half=7.074; "
        "NE2=4.607, half=-2.304 (statistical estimates, not hidden truth). "
        "Previously measured NE alternatives were less factual-efficient. "
        "Installed 0.90*joint268 delta: I_scale=.684, I_blend=.60, NE_scale=.36. "
        "Saved joint274.npz. Only core.b_in.weight edited; no gradients or optimizer. "
        "Unaccepted pending paired factual validation across tuning cohorts. "
        "Python runs: 274; forwards: 35.\n")
```