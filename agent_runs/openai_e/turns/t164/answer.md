The unchanged-dose contrasts are zero, and a fresh admission projection has exactly rank four. Those checks support using the fresh features, but the old sensitivity map remains unexplained. I’ll recompute sensitivities directly from the original weights and exact coda anchors. This avoids carrying the archive mismatch into the regression.

```python
import numpy as np
import torch
from pathlib import Path

torch.set_num_threads(8)
anchor_x143 = torch.as_tensor(arr141(precoda134), dtype=torch.float32)
anchor_h143, anchor_o143 = coda_input134(anchor_x143)

def extract_readout143(result):
    parts = list(result) if isinstance(result, (tuple, list)) else [result]
    arrays = [arr141(x) for x in parts]
    means = [x for x in arrays if x.shape == (len(head_meta),)]
    bins = [x for x in arrays if x.shape == (len(head_meta), 17)]
    assert len(means) == 1 and len(bins) >= 1, [x.shape for x in arrays]
    probabilities = [
        x for x in bins
        if np.min(x) >= -1e-7 and np.max(np.abs(x.sum(axis=1) - 1)) < 1e-4
    ]
    logits = [x for x in bins if not any(x is p for p in probabilities)]
    assert len(logits) == 1, [x.shape for x in arrays]
    return means[0].astype(np.float64), logits[0].astype(np.float64)

base_result143 = readout125(anchor_h143, anchor_o143)
print("readout component shapes",
      [tuple(x.shape) for x in base_result143] if isinstance(base_result143, (tuple, list))
      else tuple(base_result143.shape))
base_mean143, base_logits143 = extract_readout143(base_result143)
print("anchor mean agreement",
      float(np.max(np.abs(base_mean143 - arr141(mean_base136)))))

Pdir143 = Cout141 @ (delta141.reshape(-1, 1) * Bdir141)
Jm143 = np.empty((len(head_meta), rank141), dtype=np.float64)
Jl143 = np.empty((len(head_meta), 17, rank141), dtype=np.float64)
eps143 = 0.1
for j143 in range(rank141):
    shift143 = torch.as_tensor(
        (eps143 * Pdir143[:, j143]).reshape(1, 1, 192), dtype=torch.float32
    )
    hp143, op143 = coda_input134(anchor_x143 + shift143)
    hm143, om143 = coda_input134(anchor_x143 - shift143)
    mp143, lp143 = extract_readout143(readout125(hp143, op143))
    mm143, lm143 = extract_readout143(readout125(hm143, om143))
    Jm143[:, j143] = (mp143 - mm143) / (2 * eps143)
    Jl143[:, :, j143] = (lp143 - lm143) / (2 * eps143)

prob143 = np.exp(base_logits143 - base_logits143.max(axis=1, keepdims=True))
prob143 /= prob143.sum(axis=1, keepdims=True)
centred_J143 = Jl143 - np.einsum("rb,rbk->rk", prob143, Jl143)[:, None, :]
fisher143 = np.einsum("rbk,rb,rbl->rkl", centred_J143, prob143, centred_J143)
print("recomputed mean-Jacobian RMS",
      float(np.sqrt(np.mean(Jm143**2))))
print("change from archived mean Jacobian",
      float(np.sqrt(np.mean((Jm143 - Jm141)**2))))
print("Fisher eigenvalue range",
      np.round(np.quantile(np.linalg.eigvalsh(fisher143), [0, .5, 1]), 9))
print("fresh admission-null residual",
      float(np.sqrt(np.mean((admission142 @ Binput142)**2))))

with open("/work/notes.md", "a") as f143:
    f143.write(
        "\n## Run 143: fresh weight-mapped output sensitivities\n"
        "Historical XF95 and prebasis136 discrepancies were not explained; they are not "
        "used in the new regression. Recomputed 14 central finite-difference reading "
        "sensitivities using original softplus(delta_in), c_out, exact GPU coda anchors, "
        "and fresh admission-null features. No gradients and no weights changed. "
        "Python runs: 143; forwards: 15.\n"
    )
```