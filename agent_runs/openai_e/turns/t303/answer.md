The norepinephrine analysis supports roughly +4.2–4.6 mmHg for doubling and −2.1–2.3 for halving on the previously examined schedules. My last detector check mixed raw and standardized coefficients, so its large factual RMS was not meaningful. I’ll verify that mapping and install a small trial; low raw-feature factual output and correctly signed dose contrasts are prerequisites.

```python
import numpy as np
from pathlib import Path

# The saved coefficient acts on RAW GELU features, not standardized X244.
raw_design262 = np.column_stack([flat244, np.ones(len(flat244))])
fact_pred262 = raw_design262 @ coef261
std_coef262 = coef261.copy()
std_coef262[:-1] *= scale244
projected_coef262 = B244.T @ std_coef262
mapped_pred262 = F244 @ projected_coef262

print("raw factual RMS", float(np.sqrt(np.mean(fact_pred262**2))))
print("raw/projected max discrepancy",
      float(np.max(np.abs(fact_pred262 - mapped_pred262))))
print("initial-admission max output",
      float(np.max(np.abs(fact_pred262[initial_mask244]))))

contrast_mean262 = {}
for _k in D244:
    _p = D244[_k] @ projected_coef262
    _y = Y244[_k]
    contrast_mean262[_k] = float(np.mean(_p))
    print(_k, "prediction mean/rms", float(np.mean(_p)),
          float(np.sqrt(np.mean(_p**2))),
          "slope", float(np.dot(_p, _y) / np.dot(_y, _y)))

assert np.sqrt(np.mean(fact_pred262**2)) < 0.04
assert np.max(np.abs(fact_pred262 - mapped_pred262)) < 1e-5
assert contrast_mean262[("norepinephrine", 2.0)] > 0
assert contrast_mean262[("norepinephrine", 0.5)] < 0

# Conservative trial: mean local MAP increment of 0.20 mmHg for NE doubling.
scale262 = 0.20 / contrast_mean262[("norepinephrine", 2.0)]
delta_w262 = scale262 * np.outer(direction261, coef261[:-1])
delta_b262 = scale262 * direction261 * coef261[-1]
save_edit("core.prelude.0.ffn.2.weight",
          np.asarray(load_original("core.prelude.0.ffn.2.weight")) +
          delta_w262.astype(np.float32))
save_edit("core.prelude.0.ffn.2.bias",
          np.asarray(load_original("core.prelude.0.ffn.2.bias")) +
          delta_b262.astype(np.float32))
np.savez("/work/nonlinear262.npz", scale=scale262,
         delta_w=delta_w262.astype(np.float32),
         delta_b=delta_b262.astype(np.float32),
         coefficient=coef261, direction=direction261)
print("installed scale", scale262,
      "local factual MAP RMS", scale262 * np.sqrt(np.mean(fact_pred262**2)),
      "local NE half mean", scale262 * contrast_mean262[("norepinephrine", 0.5)])
print("edited", edited())

_notes = Path("/work/notes.md").read_text()
_insulin_lines = [s for s in _notes.splitlines()
                  if "insulin" in s.lower() and "target" in s.lower()
                  and len(s) < 1500]
print("insulin target evidence", "\n".join(_insulin_lines[:7])[:1700])
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 262: first conservative nonlinear trial\n"
             f"Verified raw coefficient mapping. Added rank-one prelude FFN output weight/bias edit "
             f"at scale {scale262:.6g}, targeting +0.20 mmHg local NE-doubling response. "
             "Joint236 b_in retained. Trial is unaccepted pending paired factual and rollout tests. "
             "Python runs: 262; forwards: 29.\n")
```