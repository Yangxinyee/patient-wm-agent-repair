Even removing all context dependence barely changed the rollouts, so I’ll reject that modification. Next I’ll test a drug-specific graft: borrow the norepinephrine-direction component of the stronger retained checkpoint, while keeping the validated insulin component. It succeeds only if norepinephrine responses improve without spoiling doubled insulin or factual fit.

```python
import numpy as np
import torch

# Revert the unsuccessful flat-policy change.
for _r298_name in list(edited()):
    if _r298_name not in checkpoint294:
        drop_edit(_r298_name)
for _r298_name, _r298_tensor in checkpoint294.items():
    save_edit(_r298_name, _r298_tensor.clone())

# The two raw-state directions were calibrated by finite differences, not gradients.
_r298_D = np.asarray(dirs294, dtype=np.float64)
assert _r298_D.shape == (192, 2)
assert np.linalg.matrix_rank(_r298_D) == 2
_r298_dual = np.linalg.pinv(_r298_D)
_r298_Pne = _r298_D[:, 1:2] @ _r298_dual[1:2, :]
_r298_Pboth = _r298_D @ _r298_dual
print("Direction condition number:", float(np.linalg.cond(_r298_D)))
print("NE projector norm:", float(np.linalg.norm(_r298_Pne)))
print("Insulin-direction leakage:",
      float(np.linalg.norm(_r298_Pne @ _r298_D[:, 0])))
print("NE-direction reconstruction error:",
      float(np.linalg.norm(_r298_Pne @ _r298_D[:, 1] - _r298_D[:, 1])))

_r298_outputs = [
    "core.prelude.0.attn.out.weight",
    "core.prelude.0.ffn.2.weight",
    "core.prelude.0.ffn.2.bias",
]
ne_graft_deltas298 = {}
for _r298_name in _r298_outputs:
    _r298_base = checkpoint294[_r298_name].cpu().numpy().astype(np.float64)
    _r298_strong = checkpoint287[_r298_name].cpu().numpy().astype(np.float64)
    _r298_delta = _r298_strong - _r298_base
    _r298_graft = _r298_Pne @ _r298_delta
    ne_graft_deltas298[_r298_name] = _r298_graft.astype(np.float32)
    _r298_norm = max(float(np.linalg.norm(_r298_delta)), 1e-20)
    print(_r298_name,
          "strong-minus-base norm", float(np.linalg.norm(_r298_delta)),
          "two-direction explained norm fraction",
          float(np.linalg.norm(_r298_Pboth @ _r298_delta) / _r298_norm),
          "NE graft norm", float(np.linalg.norm(_r298_graft)))
    save_edit(_r298_name,
              torch.from_numpy((_r298_base + _r298_graft).astype(np.float32)))

checkpoint_ne298 = {k: load(k).clone() for k in edited()}
with open("/work/notes.md", "a") as _r298_f:
    _r298_f.write(
        "\nForwards 35-36: flat-policy candidate factual delta +.00187; "
        "rollout shifts (-12.342,+1.391,+.408,-1.229). Removing all "
        "context dependence did not resolve asymmetry; rejected.\n"
        "Python run 298; forwards 36. Restored checkpoint294 and installed "
        "a reversible drug-specific output graft. For attention-out and "
        "FFN-output weight/bias, added the NE-direction component of "
        "(checkpoint287 minus checkpoint294), using the dual of the two "
        "finite-difference-calibrated raw-state drug directions. Other "
        "weights unchanged. checkpoint_ne298 preserves candidate and "
        "ne_graft_deltas298 preserves increments. Requires factual and "
        "paired-rollout validation; no improvement assumed.\n"
    )
```