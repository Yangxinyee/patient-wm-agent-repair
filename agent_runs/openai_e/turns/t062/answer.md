Most of the factual loss occurs before treatment: initial glucose predictions shift upward by roughly 11–20 units, while later factual shifts are much smaller. That suggests admission-state leakage. I’ll test whether projecting the edit away from initial-bundle features removes that leakage while retaining insulin sensitivity; a large loss of drug contrast would refute this approach.

```python
import torch
import numpy as np
import pandas as pd

print("INITIALIZATION TENSORS")
for name56 in NAMES:
    if name56.startswith("core.") and (
        name56.count(".") <= 2 or any(s in name56.lower() for s in ("init", "h0"))
    ):
        print(name56, SHAPES[name56])

print("\nSTATE LAYOUT")
print(state_df.head(6).to_string(index=False))
print("state_array46:", tuple(state_array46.shape))
print("prelude_all46:", tuple(prelude_all46.shape))

tuning56 = pd.read_csv("/data/tuning.csv")
first_state56 = (
    state_df.reset_index()
    .sort_values(["stay_id", "hour", "index"])
    .groupby("stay_id", sort=False).first().reset_index()
)
first_records56 = (
    tuning56.sort_values(["stay_id", "hour"])
    .groupby("stay_id", sort=False).head(1)
    .set_index("stay_id")
    .loc[first_state56.stay_id]
)
print("first event types:", first_records56.event.value_counts().to_dict())
print("first hours:", first_records56.hour.value_counts().head().to_dict())

first_atoms56 = [
    [recon_record_atom(row56)]
    for _, row56 in first_records56.iterrows()
]
initial_e56 = recon_sets(first_atoms56, prefix="encode", empty_zero=False)
initial_z56 = state_array46[first_state56["index"].to_numpy(), -32:]
initial_h56 = torch.zeros((len(first_state56), 4, 192), dtype=torch.float32)
initial_u56 = torch.zeros((len(first_state56), 192), dtype=torch.float32)
initial_E56 = prelude46(initial_h56, initial_e56, initial_u56, initial_z56)
initial_matrix56 = initial_E56.detach().cpu().numpy().reshape(-1, 192).astype(np.float64)

delta56 = trial_b_delta.detach().cpu().numpy().astype(np.float64)
left56, singular56, right56 = np.linalg.svd(delta56, full_matrices=False)
trial_input56 = right56[0]
_, init_s56, init_v56 = np.linalg.svd(initial_matrix56, full_matrices=False)
initial_rank56 = int(np.sum(init_s56 > init_s56[0] * 1e-6))
initial_basis56 = init_v56[:initial_rank56]
projected_input56 = trial_input56 - initial_basis56.T @ (initial_basis56 @ trial_input56)
projected_delta56 = (
    singular56[0] * np.outer(left56[:, 0], projected_input56)
).astype(np.float32)

print("\nINITIAL-NULL PROJECTION")
print("initial rank:", initial_rank56, "singular values:", init_s56[:12].round(5))
print("trial rank-one fraction:", float(singular56[0]**2 / np.sum(singular56**2)))
print("initial feature RMS before/after:",
      float(np.sqrt(np.mean((initial_matrix56 @ trial_input56)**2))),
      float(np.sqrt(np.mean((initial_matrix56 @ projected_input56)**2))))
print("retained input norm:", float(np.linalg.norm(projected_input56)))
for key56, raw_delta56 in slot_deltas47.items():
    before56 = np.asarray(raw_delta56) @ trial_input56
    after56 = np.asarray(raw_delta56) @ projected_input56
    before56 = before56.mean(axis=1)
    after56 = after56.mean(axis=1)
    denom56 = float(before56 @ before56)
    print(key56, {
        "contrast_retention": float(after56 @ before56 / max(denom56, 1e-12)),
        "before_rms": float(np.sqrt(np.mean(before56**2))),
        "after_rms": float(np.sqrt(np.mean(after56**2))),
    })
factual_before56 = slot_factual47 @ trial_input56
factual_after56 = slot_factual47 @ projected_input56
print("all-slot factual feature RMS before/after:",
      float(np.sqrt(np.mean(factual_before56**2))),
      float(np.sqrt(np.mean(factual_after56**2))))
print("No weights changed.")

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 56: admission-leakage nullspace test\n"
        "Run 55 localized most trial NLL degradation to initial readings "
        "(glucose mean shifts +11/+20), not later factual forecasts. "
        "Constructed initial prelude features from original GPU posterior "
        "states and admission atoms; tested a rank-one input projection "
        "orthogonal to those features. No weights changed. "
        "Python runs: 56; forwards: 5.\n"
    )
```