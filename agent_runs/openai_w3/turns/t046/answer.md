The saved latent means barely predict dose innovations, so a direct latent projection is not promising. I’ll instead fit rate and drug-presence extractors from action vectors, and a separate expected-rate predictor from pre-dose context. Accurate held-out extraction across dose multipliers would support a gated, dose-minus-expected-dose correction.

```python
import numpy as np, torch
from sklearn.linear_model import Ridge

torch.set_num_threads(8)
r41_drug_names = ("insulin", "norepinephrine")
r41_action_idx = np.array([
    i for i, atoms in enumerate(r40_info)
    if any(row.event in r41_drug_names for row, is_end in atoms)
], dtype=int)

def repair_action_vectors(end_mode, multipliers):
    atoms, owners = [], []
    counts = np.zeros(len(r41_action_idx), dtype=np.int64)
    for local_i, global_i in enumerate(r41_action_idx):
        for row, is_end in r40_info[global_i]:
            if row.event not in r41_drug_names:
                continue
            mult = multipliers[r41_drug_names.index(row.event)]
            changed = row._replace(
                rate=float(row.rate) * mult,
                dose=float(row.dose) * mult
            )
            atoms.append(make_atom(changed, is_end=is_end, end_mode=end_mode))
            owners.append(local_i)
            counts[local_i] += 1
    atom_tensor = torch.stack(atoms)
    hidden = mlp(atom_tensor, "encode_action.mlp1")
    summed = torch.zeros((len(r41_action_idx), 192))
    summed.index_add_(0, torch.tensor(owners), hidden)
    count_tensor = torch.tensor(counts, dtype=torch.float32).clamp_min(1)
    encoded = layer_norm(
        mlp(summed / count_tensor.sqrt()[:, None] +
            linear(count_tensor.log1p()[:, None], "encode_action.size"),
            "encode_action.mlp2"),
        "encode_action.norm"
    )
    return encoded.numpy()

r41_rates = np.zeros((len(sx), 2))
r41_present = np.zeros((len(sx), 2))
for i, atoms in enumerate(r40_info):
    for row, is_end in atoms:
        if not is_end and row.event in r41_drug_names:
            j = r41_drug_names.index(row.event)
            r41_rates[i, j] += float(row.rate)
            r41_present[i, j] = 1

r41_grid = [(1, 1), (0, 1), (.5, 1), (1.5, 1), (2, 1), (3, 1),
            (1, 0), (1, .5), (1, 1.5), (1, 2), (1, 3)]
r41_Xparts, r41_Yparts, r41_splitparts, r41_weights = [], [], [], []
for mode in ("clear_numeric", "clear_all", "preserve"):
    for multipliers in r41_grid:
        r41_Xparts.append(repair_action_vectors(mode, multipliers))
        r41_Yparts.append(np.column_stack([
            r41_rates[r41_action_idx] * np.array(multipliers),
            r41_present[r41_action_idx]
        ]))
        r41_splitparts.append(r40_train[r41_action_idx])
        r41_weights.append(np.full(len(r41_action_idx), 5.0 if multipliers == (1, 1) else 1.0))

r41_X = np.concatenate(r41_Xparts).astype(np.float64)
r41_Y = np.concatenate(r41_Yparts)
r41_fitmask = np.concatenate(r41_splitparts)
r41_w = np.concatenate(r41_weights)
r41_uscale = np.maximum(r41_X.std(axis=0), 1e-3)
r41_action_fit = Ridge(alpha=0.1, fit_intercept=False).fit(
    (r41_X / r41_uscale)[r41_fitmask], r41_Y[r41_fitmask],
    sample_weight=r41_w[r41_fitmask]
)
r41_action_coef = r41_action_fit.coef_ / r41_uscale[None, :]
r41_pred = r41_X @ r41_action_coef.T
print("=== HELD-OUT ACTION EXTRACTORS ===")
print("RMSE [insulin rate, NE rate, insulin presence, NE presence]:",
      np.sqrt(np.mean((r41_pred[~r41_fitmask] - r41_Y[~r41_fitmask]) ** 2, axis=0)))
print("Presence prediction quantiles:")
for j in range(2):
    for present in (0, 1):
        mask = (~r41_fitmask) & (r41_Y[:, 2+j] == present)
        print(r41_drug_names[j], present,
              np.quantile(r41_pred[mask, 2+j], [.01, .5, .99]))

print("\n=== EXPECTED-RATE CONTEXT FITS ===")
r41_ctx = np.zeros((len(sx), 192))
r41_ctx[r40_idx] = r40_context
r41_policies = {}
for j, drug in enumerate(r41_drug_names):
    active = r40_presence[:, j]
    expected = r41_rates[:, j] - r40_Y[:, j]
    mean = r41_ctx[active & r40_train].mean(axis=0)
    scale = np.maximum(r41_ctx[active & r40_train].std(axis=0), 1e-3)
    best = None
    for alpha in (1.0, 10.0, 100.0):
        fit = Ridge(alpha=alpha).fit(
            ((r41_ctx - mean) / scale)[active & r40_train],
            expected[active & r40_train]
        )
        prediction = fit.predict((r41_ctx - mean) / scale)
        test = active & r40_valid
        rmse = float(np.sqrt(np.mean((prediction[test] - expected[test]) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, alpha, fit)
    rmse, alpha, fit = best
    coef = fit.coef_ / scale
    bias = float(fit.intercept_ - coef @ mean)
    r41_policies[drug] = (coef, bias)
    prediction = r41_ctx @ coef + bias
    print(drug, "alpha", alpha, "expected-rate RMSE", rmse,
          "factual innovation RMS", float(np.sqrt(np.mean(
              (r41_rates[active & r40_valid, j] - prediction[active & r40_valid]) ** 2
          ))))

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 41; forwards 3. Linear extraction of dose innovations from posterior z failed (held-out R2 approximately zero). "
        "Fitting direct action-vector rate/presence extractors, augmented at x0/.5/1/1.5/2/3 and three END conventions, plus pre-dose-context expected-rate regressions. "
        "Proposed gated GELU-pair correction is zero without a WORLD drug atom and proportional to rate minus contextual expected rate when present. No edits yet.\n"
    )
```