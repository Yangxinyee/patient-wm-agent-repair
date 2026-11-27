The retained estimates put the four target shifts near −12.96, +6.48, +4.28 and −2.14. I’ll test a stronger version of the feedback hypothesis: make the two policy rows context-independent, centered on their active-drug training contexts, while leaving the dose decoders and gates unchanged. Improved rollout symmetry with acceptable factual NLL would support this change; otherwise I’ll reject it.

```python
import numpy as np
import torch

# Start from the validated alpha=.01 checkpoint, not the local projection.
for _r297_name in list(edited()):
    if _r297_name not in checkpoint294:
        drop_edit(_r297_name)
for _r297_name, _r297_tensor in checkpoint294.items():
    save_edit(_r297_name, _r297_tensor.clone())

_r297_w = load("z_post.0.weight").clone()
_r297_b = load("z_post.0.bias").clone()
_r297_ctx = np.asarray(ctx_data294, dtype=np.float64)
_r297_dose = np.asarray(dose_data294, dtype=np.float64)

for _r297_drug, _r297_row in [(0, 31), (1, 93)]:
    _r297_active = _r297_dose[:, _r297_drug] > 0
    if not np.any(_r297_active):
        raise RuntimeError("No active-drug contexts for centering.")
    _r297_mean = _r297_ctx[_r297_active].mean(axis=0)
    _r297_old = _r297_w[_r297_row, :192].cpu().numpy().astype(np.float64)
    _r297_offset = float(_r297_old @ _r297_mean)
    _r297_change = -(_r297_ctx @ _r297_old) + _r297_offset
    _r297_b[_r297_row] += _r297_offset
    _r297_w[_r297_row, :192] = 0
    print("Drug", _r297_drug, "policy row", _r297_row,
          "removed context norm", float(np.linalg.norm(_r297_old)),
          "centering offset", _r297_offset,
          "active-context activation-change RMS",
          float(np.sqrt(np.mean(_r297_change[_r297_active] ** 2))),
          "all-context RMS",
          float(np.sqrt(np.mean(_r297_change ** 2))))

save_edit("z_post.0.weight", _r297_w)
save_edit("z_post.0.bias", _r297_b)
checkpoint_flat297 = {k: load(k).clone() for k in edited()}

print("=== CARRIER OUTPUT MAPPING (FOR POSSIBLE DRUG-SPECIFIC TEST) ===")
_r297_post2 = load("z_post.2.weight").cpu().numpy()
print("z_post.2.weight shape", _r297_post2.shape)
for _r297_col in [31, 93]:
    _r297_idx = np.argsort(np.abs(_r297_post2[:, _r297_col]))[-6:][::-1]
    print("column", _r297_col,
          [(int(i), float(_r297_post2[i, _r297_col])) for i in _r297_idx])
print("Dose-direction norms", np.linalg.norm(np.asarray(dirs294), axis=0))

print("=== EARLY IDENTIFICATION NOTES ===")
print(Path("/work/notes.md").read_text()[:3300])

with open("/work/notes.md", "a") as _r297_f:
    _r297_f.write(
        "\nPython run 297; forwards 34. Installed reversible flat-policy "
        "candidate from checkpoint294 (validated alpha=.01). Zeroed "
        "context columns of posterior policy rows31/93 and adjusted biases "
        "to preserve their active-drug training-context mean activations. "
        "Dose decoders, gates, and all other weights unchanged. "
        "checkpoint_flat297 preserves candidate; checkpoint296 and "
        "checkpoint294 preserve validated fallbacks. Factual and rollout "
        "tests required; no improvement assumed.\n"
    )
```