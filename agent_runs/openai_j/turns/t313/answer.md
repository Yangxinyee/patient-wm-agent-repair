Training contains 100 norepinephrine rates up to 1.0, while insulin reaches 14.5. Shared bin edits therefore would not preserve factual behavior. I’ll instead fit a constrained action-encoder correction that smoothly extrapolates above the norepinephrine ceiling. Small factual residuals and moderate weight changes would justify testing it; large residuals or unstable coefficients would reject it.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 273: constrained action-extrapolation screen\n"
        "- Training support: insulin 26 rate/dose pairs, maximum 14.5; "
        "NE 100 pairs, maximum rate 1.0.\n"
        "- Fitting a closed-form action-MLP correction, preserving factual "
        "atoms and within-support intervention atoms. No edits installed.\n"
        "- Counts: 273 Python runs, 34 forward passes.\n"
    )

norms273 = {}
for value in VOCAB.values():
    if isinstance(value, list):
        for row in value:
            if isinstance(row, dict) and all(
                k in row for k in ("code", "attribute", "median", "iqr",
                                  "fine_bin_edges")
            ):
                norms273[(row["code"], row["attribute"])] = row
assert ("norepinephrine", "rate") in norms273

num273 = np.asarray(load_original("embed.numeric.weight"), dtype=np.float64)
bins273 = {
    a: np.asarray(load_original(f"embed.bins.{a}.weight"), dtype=np.float64)
    for a in ("dose", "rate")
}
av273 = post_av0.detach().cpu().numpy().astype(np.float64)
ra273 = repair_atoms.reset_index(drop=True)

def numeric_terms273(drug, rates, doses, side):
    rates, doses = np.asarray(rates), np.asarray(doses)
    result = np.zeros((len(rates), 192), dtype=np.float64)
    for attr, values, column in (("dose", doses, 2), ("rate", rates, 4)):
        table = norms273[(drug, attr)]
        z = np.clip((values - table["median"]) / table["iqr"], -10, 10)
        b = 1 + np.searchsorted(table["fine_bin_edges"], values, side=side)
        result += z[:, None] * num273[:, column] + bins273[attr][b]
    return result

anchors273 = {}
sides273 = {}
for drug in ("insulin", "norepinephrine"):
    ix = np.flatnonzero(
        ra273.event.eq(drug).to_numpy() & ~ra273.is_end.to_numpy()
    )
    rates = ra273.loc[ix, "rate"].to_numpy(float)
    doses = ra273.loc[ix, "dose"].to_numpy(float)
    comparisons = []
    for side in ("left", "right"):
        terms = numeric_terms273(drug, rates, doses, side)
        anchor = av273[ix[0]] - terms[0]
        error = np.max(np.abs(anchor[None, :] + terms - av273[ix]))
        comparisons.append((error, side, anchor))
    error, side, anchor = min(comparisons, key=lambda x: x[0])
    print(drug, "cached reconstruction max error:", error, "bin side:", side)
    assert error < 2e-4
    anchors273[drug], sides273[drug] = anchor, side

def action_atoms273(drug, rates):
    rates = np.asarray(rates, dtype=np.float64)
    doses = rates if drug == "insulin" else 4.8 * rates
    return anchors273[drug][None, :] + numeric_terms273(
        drug, rates, doses, sides273[drug]
    )

constraints273 = [
    action_atoms273("insulin", np.round(np.arange(0, 15.0001, .25), 4)),
    action_atoms273("norepinephrine", np.round(np.arange(0, 1.0001, .005), 4))
]

# Include exact cached factual vectors, plus every training-support start atom.
action_mask273 = ra273.event.isin(["insulin", "norepinephrine"]).to_numpy()
constraints273.append(np.unique(av273[action_mask273], axis=0))
for drug in ("insulin", "norepinephrine"):
    pairs = sorted(support272[drug])
    rates = np.array([p[0] for p in pairs])
    doses = np.array([p[1] for p in pairs])
    constraints273.append(
        anchors273[drug][None, :] +
        numeric_terms273(drug, rates, doses, sides273[drug])
    )
    # END atoms either omit numbers or retain the infusion's numbers.
    end_ix = np.flatnonzero(
        ra273.event.eq(drug).to_numpy() & ra273.is_end.to_numpy()
    )
    if len(end_ix):
        observed_end = av273[end_ix]
        if np.max(np.abs(observed_end - observed_end[0])) > 2e-4:
            r0 = ra273.loc[end_ix[0], "rate"]
            d0 = ra273.loc[end_ix[0], "dose"]
            end_anchor = observed_end[0] - numeric_terms273(
                drug, np.array([r0]), np.array([d0]), sides273[drug]
            )[0]
            reconstructed_end = end_anchor + numeric_terms273(
                drug, ra273.loc[end_ix, "rate"].to_numpy(float),
                ra273.loc[end_ix, "dose"].to_numpy(float), sides273[drug]
            )
            end_error = np.max(np.abs(reconstructed_end - observed_end))
            print(drug, "END reconstruction error:", end_error)
            assert end_error < 2e-4
            constraints273.append(
                end_anchor + numeric_terms273(drug, rates, doses, sides273[drug])
            )

atom_constraints273 = np.unique(np.vstack(constraints273), axis=0)
w10_273 = torch.as_tensor(load_original("encode_action.mlp1.0.weight"))
b10_273 = torch.as_tensor(load_original("encode_action.mlp1.0.bias"))
w12_273 = np.asarray(load_original("encode_action.mlp1.2.weight"),
                    dtype=np.float64)
b12_273 = np.asarray(load_original("encode_action.mlp1.2.bias"),
                    dtype=np.float64)

activation_names273 = set(mlp.__code__.co_names)
print("MLP activation references:", activation_names273)
assert any(a in activation_names273 for a in ("gelu", "silu", "relu"))
activation273 = (
    F.gelu if "gelu" in activation_names273 else
    F.silu if "silu" in activation_names273 else F.relu
)

def hidden_atoms273(atoms):
    x = torch.as_tensor(atoms, dtype=torch.float32)
    h = activation273(F.linear(x, w10_273, b10_273))
    return np.column_stack((h.numpy().astype(np.float64), np.ones(len(atoms))))

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
upper_hidden273 = hidden_atoms273(action_atoms273("norepinephrine", upper_rates273))
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
print("off-support feature singular values:", sn273[:8])
print("target correction RMS:", np.sqrt(np.mean(target_delta273 ** 2)))

atom_extrap_betas273 = {}
rows273 = []
for ridge in (.01, .1, 1., 10., 100.):
    beta = vhn273.T @ (
        (sn273 / (sn273 ** 2 + ridge))[:, None] *
        (un273.T @ target_delta273)
    )
    beta -= rowspace273.T @ (rowspace273 @ beta)
    atom_extrap_betas273[ridge] = beta
    factual_delta = constraint_hidden273 @ beta
    residual = off_hidden273 @ beta - target_delta273
    rows273.append({
        "ridge": ridge,
        "factual_max": np.max(np.abs(factual_delta)),
        "factual_rms": np.sqrt(np.mean(factual_delta ** 2)),
        "target_residual_rms": np.sqrt(np.mean(residual ** 2)),
        "weight_delta_norm": np.linalg.norm(beta[:-1]),
        "relative_weight_norm": np.linalg.norm(beta[:-1]) / np.linalg.norm(w12_273),
    })
frontier273 = pd.DataFrame(rows273)
print(frontier273.to_string(index=False))
frontier273.to_csv("/work/action_extrapolation_frontier273.csv", index=False)
print("NO WEIGHTS CHANGED:", edited())
```