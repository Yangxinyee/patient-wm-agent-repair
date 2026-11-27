The factual atom reconstruction matched the cache within \(5\times10^{-7}\), but the fit stopped because the activation is supplied through a helper. No weights changed. I’ll use that helper and verify the full MLP reconstruction before completing the fit; disagreement would invalidate the correction.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

activation273 = globals()["activation"]
assert callable(activation273)
print("Activation helper:", repr(activation273)[:300])

def hidden_atoms273(atoms):
    x = torch.as_tensor(atoms, dtype=torch.float32)
    h = activation273(F.linear(x, w10_273, b10_273))
    return np.column_stack((h.numpy().astype(np.float64), np.ones(len(atoms))))

check_atoms274 = torch.as_tensor(atom_constraints273[:12], dtype=torch.float32)
params274 = list(inspect.signature(mlp).parameters)
print("MLP parameters:", params274)
if params274[0] in ("p", "prefix", "name"):
    helper_outputs274 = mlp("encode_action.mlp1", check_atoms274)
else:
    helper_outputs274 = mlp(check_atoms274, "encode_action.mlp1")

check_hidden274 = hidden_atoms273(check_atoms274.numpy())
reconstructed274 = check_hidden274[:, :-1] @ w12_273.T + b12_273
mlp_error274 = np.max(np.abs(
    reconstructed274 - helper_outputs274.detach().cpu().numpy()
))
print("Full MLP reconstruction max error:", mlp_error274)
assert mlp_error274 < 2e-4

constraint_hidden273 = hidden_atoms273(atom_constraints273)
_, s_constraint273, vh_constraint273 = np.linalg.svd(
    constraint_hidden273, full_matrices=False
)
rank273 = int(np.sum(s_constraint273 > s_constraint273[0] * 1e-7))
rowspace273 = vh_constraint273[:rank273]
print("constraints:", constraint_hidden273.shape, "numerical rank:", rank273)

off_rates273 = np.round(np.arange(1.005, 3.0001, .005), 4)
off_hidden273 = hidden_atoms273(action_atoms273("norepinephrine", off_rates273))
off_original273 = off_hidden273[:, :-1] @ w12_273.T + b12_273

upper_rates273 = np.round(np.arange(.88, 1.0001, .01), 4)
upper_hidden273 = hidden_atoms273(
    action_atoms273("norepinephrine", upper_rates273)
)
upper_outputs273 = upper_hidden273[:, :-1] @ w12_273.T + b12_273
centered273 = upper_rates273 - upper_rates273.mean()
slope273 = centered273 @ upper_outputs273 / (centered273 @ centered273)
endpoint273 = upper_outputs273[-1]
target_outputs273 = (
    endpoint273[None, :] + (off_rates273 - 1)[:, None] * slope273
)
target_delta273 = target_outputs273 - off_original273

null_features273 = off_hidden273 - (
    off_hidden273 @ rowspace273.T
) @ rowspace273
un273, sn273, vhn273 = np.linalg.svd(null_features273, full_matrices=False)
print("off-support singular values:", sn273[:8])
print("target correction RMS:", np.sqrt(np.mean(target_delta273 ** 2)))

atom_extrap_betas273 = {}
rows274 = []
for ridge in (.01, .1, 1., 10., 100.):
    beta = vhn273.T @ (
        (sn273 / (sn273 ** 2 + ridge))[:, None] *
        (un273.T @ target_delta273)
    )
    beta -= rowspace273.T @ (rowspace273 @ beta)
    atom_extrap_betas273[ridge] = beta
    factual_delta = constraint_hidden273 @ beta
    residual = off_hidden273 @ beta - target_delta273
    rows274.append({
        "ridge": ridge,
        "factual_max": np.max(np.abs(factual_delta)),
        "factual_rms": np.sqrt(np.mean(factual_delta ** 2)),
        "target_residual_rms": np.sqrt(np.mean(residual ** 2)),
        "weight_delta_norm": np.linalg.norm(beta[:-1]),
        "relative_weight_norm": np.linalg.norm(beta[:-1]) / np.linalg.norm(w12_273),
    })

frontier273 = pd.DataFrame(rows274)
print(frontier273.to_string(index=False))
frontier273.to_csv("/work/action_extrapolation_frontier273.csv", index=False)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 274: corrected activation and nullspace fit\n"
        f"- Factual atom reconstruction <5e-7; full MLP reconstruction "
        f"error {mlp_error274:.8g}.\n"
        f"- Constraint rank {rank273}; statistical extrapolation frontier "
        "saved to action_extrapolation_frontier273.csv.\n"
        "- No weights changed. Requires local-response and GPU validation.\n"
        "- Counts: 274 Python runs, 34 forward passes.\n"
    )
print("CURRENT EDITS:", edited())
```