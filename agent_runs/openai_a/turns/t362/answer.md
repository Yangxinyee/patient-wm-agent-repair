The insulin threshold estimates remain negative across the tested specifications, supporting the correction’s direction. Their sustained-effect estimate depends on identification and dynamic-model assumptions; it is not observed counterfactual truth. I’ll retain the validated edit and record the final metrics. A checkpoint mismatch or non-finite weight would block submission.

```python
from pathlib import Path
import numpy as np

_expected298 = {
    "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.0.weight",
    "attr_heads.value_bin.2.bias",
    "attr_heads.value_bin.2.weight",
}
assert set(edited()) == _expected298
_checkpoint298 = globals()["amplitude093_checkpoint"]

print("=== FINAL WEIGHT INTEGRITY ===")
for name in sorted(_expected298):
    current = np.asarray(load(name))
    original = np.asarray(load_original(name))
    reference = np.asarray(_checkpoint298[name])
    assert current.shape == tuple(SHAPES[name])
    assert np.isfinite(current).all()
    assert np.array_equal(current, reference)
    print(name, "validated checkpoint match;",
          int(np.count_nonzero(current != original)), "modified entries")

_final_note298 = """
## Final submission audit; Python 298, forwards 50/50
- Retaining amplitude093_checkpoint unchanged. Exactly four value-bin-head
  tensors are edited; all are finite, correctly shaped, and exactly match
  the GPU-validated checkpoint. No further weight changes.
- The correction uses two gated hidden units (41,109), with sharp240 activity
  gating and output-graft amplitude0.93. No NE-specific correction was made.
- Insulin dosing-threshold estimates are consistently negative across tested
  bandwidths and polynomial specifications. Fitted beta=-1.928709 and
  persistence=0.831840 imply a training-record sustained x2 target=-11.136162
  mg/dL and half target=+5.568081 mg/dL. These are estimates conditional on
  threshold identification and dynamic-response assumptions, not observed truth.
- NE evidence remains weak or specification-sensitive. Leaving its response
  essentially unchanged is conservative, not a conclusion that its true effect
  is absent.
- Validated pooled tuning rollout predictions, over reached target stay-hours:
  insulin_x2=-8.792596 mg/dL; insulin_half=+4.087351 mg/dL;
  ne_x2=-0.226430 mmHg; ne_half=-0.558487 mmHg.
- Full400 tuning factual NLL increase=0.00336585 nats/reading, 18654 readings;
  stay-bootstrap95% interval=[0.00204746,0.00474793].
- Additional prespecified200 training stays: increase=0.00244200, 9520 readings;
  stay-bootstrap95% interval=[0.00039733,0.00461895].
- Both measured factual checks pass the +0.005 limit. Unseen-stay compliance
  remains an inference, not a guarantee.
"""
with Path("/work/notes.md").open("a") as f:
    f.write(_final_note298)

print("\nFinal notebook entry saved. No weights changed.")
print(_final_note298)
```
