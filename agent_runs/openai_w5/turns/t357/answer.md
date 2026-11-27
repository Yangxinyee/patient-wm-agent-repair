The exact audit supports undoing the intercept calibration, which is now reverted. The earlier head diagnostics also place factual signal units well above the GELU cutoff, so a shared-offset edit is not supported. I’ll instead test 1,100-second insulin persistence with interpolated factual centering. Lower pooled loss with substantial dose responses would support it; otherwise I’ll restore the validated 1,250-second baseline.

```python
import torch, numpy as np
from safetensors.torch import load_file, save_file
from pathlib import Path

CP900299 = load_file("/work/checkpoint_feasible900.safetensors")
tau299 = 1100.0
alpha299 = (tau299 - 900.0) / (1250.0 - 900.0)
glucose_rows299 = [74, 171, 107, 7]
private_cols299 = [21, 61]

print("=== REVERSIBLE PERSISTENCE TRIAL ===")
print("target_tau_seconds", tau299, "centering_interpolation", alpha299)
differences299 = []
for n in BASE_FINAL298:
    if n in CP900299 and not torch.equal(BASE_FINAL298[n], CP900299[n]):
        differences299.append((n, float((BASE_FINAL298[n]-CP900299[n]).abs().max())))
print("900_TO_1250_CHECKPOINT_DIFFERENCES", differences299)

flow299 = load("core.flow.a").clone()
print("previous_private_taus",
      torch.exp(-flow299[private_cols299]).tolist())
flow299[private_cols299] = float(np.log(1.0 / tau299))

new_heads299 = {}
for n in ("attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias"):
    current = BASE_FINAL298[n].clone()
    low = CP900299[n]
    proposed = current.clone()
    for row in glucose_rows299:
        proposed[row] = low[row] + alpha299 * (current[row] - low[row])
    if proposed.ndim == 2:
        # Drug-response coefficients are fixed; interpolate only factual centering.
        proposed[:, private_cols299] = current[:, private_cols299]
    new_heads299[n] = proposed
    print(n, "max_change", float((proposed-current).abs().max()),
          "changed_rows",
          torch.nonzero((proposed-current).abs().reshape(proposed.shape[0], -1)
                        .max(1).values > 0).flatten().tolist())

# All checks precede installation; no drug strength or NE transport is changed.
assert torch.isfinite(flow299).all()
assert all(torch.isfinite(t).all() for t in new_heads299.values())
assert torch.equal(
    new_heads299["attr_heads.value_bin.0.weight"][:, private_cols299],
    BASE_FINAL298["attr_heads.value_bin.0.weight"][:, private_cols299])
save_edit("core.flow.a", flow299)
for n, t in new_heads299.items():
    save_edit(n, t.contiguous())

BASE_TRIAL299 = {n: load(n).clone().contiguous() for n in edited()}
save_file(BASE_TRIAL299, "/work/checkpoint_trial299.safetensors")
print("installed_private_taus",
      torch.exp(-load("core.flow.a")[private_cols299]).tolist())
print("Edited tensor count", len(edited()))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 299: final reversible persistence/centering trial\n"
            "- Rejected shared-offset idea: earlier reconstructed factual signal "
            "units are near +8, not demonstrably GELU-clipped.\n"
            "- Installed insulin private-coordinate tau=1100 seconds, with factual "
            f"centering interpolated {alpha299:.8f} from the saved 900-second "
            "candidate toward the validated 1250-second baseline.\n"
            "- Preserved private drug coefficients, output-head strengths, NE "
            "transport, and every unrelated tensor.\n"
            "- Trial saved in /work/checkpoint_trial299.safetensors; rollback snapshot "
            "is /work/checkpoint_baseline298.safetensors.\n"
            "- Acceptance requires full 400-stay paired factual testing and paired "
            "dose rollouts. Seek lower loss than +0.00414485 without losing "
            "substantial dose responses; otherwise restore the exact baseline.\n"
            "- Python runs: 299/300. Forward passes: 42/50. One Python run is "
            "reserved for final audit, notebook update, and conditional rollback.\n")
```
