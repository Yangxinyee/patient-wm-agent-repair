The cached screen predicts only a small factual-NLL increase for norepinephrine 0.40 versus 0.35, and the safe checkpoint matches its backup exactly. That warrants a reversible trial, not acceptance. I’ll install the candidate after verifying its reconstruction, then validate all 400 factual stays. A conservative upper bound below +0.005, followed by improved rollout effects, would justify keeping it.

```python
import numpy as np
import torch
from pathlib import Path

def _array274(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

r274_safe = {k: _array274(load(k)).copy() for k in edited()}
r274_old_head, r274_old_table = n262_candidates[(0.90, 0.35, 4.0)]
r274_new_head, r274_new_table = n262_candidates[(0.90, 0.40, 4.0)]
r274_transition_name = "embed_transition.value_bin.weight"
r274_original_transition = _array274(
    load_original(r274_transition_name)
).copy()

def _transition274(table, sign):
    out = r274_original_transition.copy()
    for (event, last_bin), correction in table.items():
        row = int(r257_codes[event]) * 17 + int(last_bin)
        out[row] += np.float32(sign) * _array274(correction)
    return out

r274_head_error = max(
    float(np.max(np.abs(_array274(v) - r274_safe[k])))
    for k, v in r274_old_head.items()
)
r274_sign_errors = {
    sign: float(np.max(np.abs(
        _transition274(r274_old_table, sign) -
        r274_safe[r274_transition_name]
    )))
    for sign in (-1.0, 1.0)
}
r274_sign = min(r274_sign_errors, key=r274_sign_errors.get)
print("safe-head reconstruction error:", r274_head_error)
print("safe-transition sign errors:", r274_sign_errors)
assert r274_head_error < 2e-6
assert r274_sign_errors[r274_sign] < 2e-6

r274_trial = {k: _array274(v).copy()
              for k, v in r274_new_head.items()}
r274_trial[r274_transition_name] = _transition274(
    r274_new_table, r274_sign
)
assert set(r274_trial) == set(r274_safe)
assert all(np.isfinite(v).all() for v in r274_trial.values())

np.savez("/work/safe_before_ne40.npz", **r274_safe)
np.savez("/work/candidate90_40_bin80x4.npz", **r274_trial)
try:
    for _k, _v in r274_trial.items():
        save_edit(_k, torch.as_tensor(_v, dtype=torch.float32))
except Exception:
    for _k, _v in r274_safe.items():
        save_edit(_k, torch.as_tensor(_v, dtype=torch.float32))
    raise

print("installed insulin .90 / NE .40; cancellation shrink80-scale4")
print("changed tensor norms relative to safe checkpoint:")
for _k in sorted(r274_trial):
    print(_k, float(np.linalg.norm(r274_trial[_k] - r274_safe[_k])))

with Path("/work/notes.md").open("a") as _f:
    _f.write(
        "\n## Run 274: reversible small stronger-NE trial installed\n"
        "Verified exact reconstruction of safe .90/.35 checkpoint before "
        "installing insulin .90 / NE .40 with shrink80-scale4 cancellation. "
        "All coefficients remain training-record fitted. Backup: "
        "/work/safe_before_ne40.npz; trial: "
        "/work/candidate90_40_bin80x4.npz. Not accepted yet. Require all400 "
        "paired factual upper95 < .005 and improved exact-checkpoint rollout; "
        "otherwise restore the safe checkpoint.\n"
    )
```
