The cached MLP also requires an explicit activation type. No edit was attempted. I’ll extract that type directly from the encoder helper, then finish the fit. Only verified reconstruction and negligible factual perturbation would justify a rollout test.

```python
from pathlib import Path
import dis
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Runs 274–275: encoder interface correction\n"
        "- Run 274 stopped because mlp requires an explicit activation kind; "
        "no weights changed.\n"
        "- Recovering the kind directly from repair_set_encode, not guessing.\n"
        "- Counts entering run 275: 274 Python runs, 34 forward passes.\n"
    )

instructions275 = list(dis.get_instructions(repair_set_encode))
start275 = next(
    i for i, ins in enumerate(instructions275)
    if ins.opname == "LOAD_GLOBAL" and ins.argval == "mlp"
)
end275 = next(
    i for i in range(start275 + 1, len(instructions275))
    if instructions275[i].opname == "CALL"
)
window275 = instructions275[start275:end275 + 1]
print("First encoder MLP call:",
      [(i.opname, i.argval) for i in window275])
print("Activation signature:", inspect.signature(activation))
kind_candidates275 = [
    i.argval for i in window275
    if i.opname == "LOAD_CONST" and isinstance(i.argval, str)
    and i.argval.lower() in ("gelu", "silu", "relu", "tanh", "sigmoid")
]
assert len(kind_candidates275) == 1, repair_set_encode.__code__.co_consts
kind273 = kind_candidates275[0]
print("Verified encoder activation kind:", kind273)

def hidden_atoms273(atoms):
    x = torch.as_tensor(atoms, dtype=torch.float32)
    h = activation(F.linear(x, w10_273, b10_273), kind273)
    return np.column_stack((h.numpy().astype(np.float64), np.ones(len(atoms))))

check_atoms275 = torch.as_tensor(atom_constraints273[:12], dtype=torch.float32)
helper_outputs275 = mlp(check_atoms275, "encode_action.mlp1", kind273)
check_hidden275 = hidden_atoms273(check_atoms275.numpy())
reconstructed275 = check_hidden275[:, :-1] @ w12_273.T + b12_273
mlp_error275 = np.max(np.abs(
    reconstructed275 - helper_outputs275.detach().cpu().numpy()
))
print("Full MLP reconstruction max error:", mlp_error275)
assert mlp_error275 < 2e-4

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
rows275 = []
for ridge in (.01, .1, 1., 10., 100.):
    beta = vhn273.T @ (
        (sn273 / (sn273 ** 2 + ridge))[:, None] *
        (un273.T @ target_delta273)
    )
    beta -= rowspace273.T @ (rowspace273 @ beta)
    atom_extrap_betas273[ridge] = beta
    factual_delta = constraint_hidden273 @ beta
    residual = off_hidden273 @ beta - target_delta273
    rows275.append({
        "ridge": ridge,
        "factual_max": np.max(np.abs(factual_delta)),
        "factual_rms": np.sqrt(np.mean(factual_delta ** 2)),
        "target_residual_rms": np.sqrt(np.mean(residual ** 2)),
        "weight_delta_norm": np.linalg.norm(beta[:-1]),
        "relative_weight_norm": np.linalg.norm(beta[:-1]) / np.linalg.norm(w12_273),
    })

frontier273 = pd.DataFrame(rows275)
print(frontier273.to_string(index=False))
frontier273.to_csv("/work/action_extrapolation_frontier273.csv", index=False)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 275: verified constrained extrapolation fit\n"
        f"- Encoder activation {kind273}; full MLP reconstruction "
        f"error {mlp_error275:.8g}.\n"
        f"- Constraint rank {rank273}; frontier saved.\n"
        "- No weights changed; local response and GPU validation still needed.\n"
        "- Counts: 275 Python runs, 34 forward passes.\n"
    )
print("CURRENT EDITS:", edited())
```