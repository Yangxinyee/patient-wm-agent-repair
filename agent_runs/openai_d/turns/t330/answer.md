The reconstruction check failed before any weights changed: the head matched exactly, but the transition correction did not. The stored table likely needs the checkpoint’s additional scale factor. I’ll recover that factor directly and require exact reconstruction before installing the trial. If it still fails, I’ll stop rather than alter the validated model incorrectly.

```python
import numpy as np
import torch
from pathlib import Path

def _table_delta275(table):
    delta = np.zeros_like(r274_original_transition, dtype=np.float32)
    for (event, last_bin), correction in table.items():
        row = int(r257_codes[event]) * 17 + int(last_bin)
        delta[row] = _array274(correction)
    return delta

r275_old_delta = _table_delta275(r274_old_table)
r275_installed_delta = (
    r274_safe[r274_transition_name] - r274_original_transition
)
r275_multiplier = float(
    np.sum(r275_old_delta.astype(np.float64) *
           r275_installed_delta.astype(np.float64)) /
    np.sum(r275_old_delta.astype(np.float64) ** 2)
)
r275_old_reconstructed = (
    r274_original_transition +
    np.float32(r275_multiplier) * r275_old_delta
)
r275_error = float(np.max(np.abs(
    r275_old_reconstructed - r274_safe[r274_transition_name]
)))
print("recovered table multiplier:", r275_multiplier)
print("safe transition reconstruction error:", r275_error)
assert r274_head_error < 2e-6
assert r275_error < 2e-6

r275_trial = {k: _array274(v).copy()
              for k, v in r274_new_head.items()}
r275_trial[r274_transition_name] = (
    r274_original_transition +
    np.float32(r275_multiplier) * _table_delta275(r274_new_table)
)
assert set(r275_trial) == set(r274_safe)
assert all(np.isfinite(v).all() for v in r275_trial.values())

np.savez("/work/safe_before_ne40.npz", **r274_safe)
np.savez("/work/candidate90_40_bin80x4.npz", **r275_trial)
try:
    for _k, _v in r275_trial.items():
        save_edit(_k, torch.as_tensor(_v, dtype=torch.float32))
except Exception:
    for _k, _v in r274_safe.items():
        save_edit(_k, torch.as_tensor(_v, dtype=torch.float32))
    raise

print("installed insulin .90 / NE .40 with reconstructed scale4 correction")
for _k in sorted(r275_trial):
    print(_k, "change_norm",
          float(np.linalg.norm(r275_trial[_k] - r274_safe[_k])))

with Path("/work/notes.md").open("a") as _f:
    _f.write(
        "\n## Runs 274–275: verified small stronger-NE trial\n"
        "Run274 stopped before changing weights because the cached table "
        "was unscaled. Run275 recovered its multiplier directly from the "
        f"safe checkpoint ({r275_multiplier:.8f}), with maximum "
        f"reconstruction error {r275_error:.3g}, then installed .90 insulin / "
        ".40 NE with the same shrink80-scale4 cancellation. Exact safe backup "
        "is /work/safe_before_ne40.npz. Trial is NOT accepted: require all400 "
        "paired factual upper95 < .005 and improved exact-checkpoint rollout; "
        "otherwise restore the safe checkpoint.\n"
    )
```
