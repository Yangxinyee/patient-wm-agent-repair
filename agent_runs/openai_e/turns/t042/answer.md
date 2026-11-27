This is the same truncated run-38 output, so it adds no new evidence. I’ll proceed with the numerical verification and contrast fit, keeping the output compact. Accurate reconstruction plus small held-out factual scores would justify testing a latent edit; poor dose contrasts would refute this feature design.

```python
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

torch.set_num_threads(8)
cf_starts = defaultdict(list)
cf_ends = defaultdict(list)
cf_ids = set(state_df.stay_id.unique())
for row in tune[tune.stay_id.isin(cf_ids) &
                tune.event.isin(["insulin", "norepinephrine"])].itertuples(index=False):
    cf_starts[(int(row.stay_id), int(round(row.hour * 60)))].append(row)
    cf_ends[(int(row.stay_id), int(round(row.end_hour * 60)))].append(row)

cf_rows = np.array([
    i for i, row in enumerate(state_df.itertuples(index=False))
    if cf_starts.get((int(row.stay_id), int(round(row.hour * 60))))
], dtype=int)
cf_frame = state_df.iloc[cf_rows].copy()
cf_e_lists, cf_a_lists = [], []
cf_doses = np.zeros((len(cf_rows), 2), dtype=np.float32)
for j, row in enumerate(cf_frame.itertuples(index=False)):
    key = (int(row.stay_id), int(round(row.hour * 60)))
    end_atoms = []
    for end in cf_ends.get(key, []):
        atom = recon_record_atom(end, kind=2, end_numeric="absent", end_units="none")
        atom += W0["embed.ekind.weight"][2] - W0["embed.ekind.weight"][RECON_EKINDS[end.event]]
        end_atoms.append(atom)
    cf_e_lists.append(end_atoms)
    starts = cf_starts[key]
    cf_a_lists.append([recon_record_atom(r) for r in starts])
    for r in starts:
        cf_doses[j, ["insulin", "norepinephrine"].index(r.event)] += r.rate

cf_E = recon_sets(cf_e_lists, prefix="encode", empty_zero=False)
cf_U = recon_sets(cf_a_lists, prefix="encode_action", empty_zero=True)
cf_C = recon_context[cf_rows]
cf_H0_t = F.gelu(F.linear(torch.cat([cf_C, cf_E, cf_U], dim=-1),
                          W0["z_post.0.weight"], W0["z_post.0.bias"]))
cf_Z0 = F.linear(cf_H0_t, W0["z_post.2.weight"], W0["z_post.2.bias"])[:, :32]
cf_truth = torch.as_tensor(S0["x"], dtype=torch.float32)[cf_rows, -32:]
cf_err = torch.sqrt(torch.mean((cf_Z0 - cf_truth) ** 2, dim=-1)).numpy()
print("Posterior verification:", {
    "rows": len(cf_rows), "RMS": float(np.sqrt(np.mean(cf_err**2))),
    "median": float(np.median(cf_err)), "max": float(cf_err.max()),
    "within_0.002": int((cf_err < .002).sum())})

cf_H0 = cf_H0_t.numpy()
cf_H_variants, cf_targets = [], []
for drug_i, drug in enumerate(["insulin", "norepinephrine"]):
    for mult in [2.0, 0.5]:
        lists = []
        for row in cf_frame.itertuples(index=False):
            key = (int(row.stay_id), int(round(row.hour * 60)))
            lists.append([
                recon_record_atom(r._replace(rate=r.rate * mult, dose=r.dose * mult))
                if r.event == drug else recon_record_atom(r)
                for r in cf_starts[key]
            ])
        Uedit = recon_sets(lists, prefix="encode_action", empty_zero=True)
        Hedit = F.gelu(F.linear(torch.cat([cf_C, cf_E, Uedit], dim=-1),
                               W0["z_post.0.weight"], W0["z_post.0.bias"])).numpy()
        target_edit = np.zeros_like(cf_doses)
        target_edit[:, drug_i] = (mult - 1) * cf_doses[:, drug_i]
        cf_H_variants.append(Hedit)
        cf_targets.append(target_edit)

cf_train = (cf_frame.stay_id.to_numpy() % 5) != 0
cf_test = ~cf_train
cf_scale = np.maximum(cf_H0[cf_train].std(axis=0), 0.05)
cf_F0 = np.column_stack([cf_H0 / cf_scale, np.ones(len(cf_H0))])
cf_D = [np.column_stack([(hh - cf_H0) / cf_scale, np.zeros(len(hh))])
        for hh in cf_H_variants]
cf_other_rows = np.setdiff1d(np.arange(len(state_df)), cf_rows)
cf_Fother = np.column_stack([
    recon_post_hidden[cf_other_rows].numpy() / cf_scale,
    np.ones(len(cf_other_rows))
])
cf_other_train = (state_df.iloc[cf_other_rows].stay_id.to_numpy() % 5) != 0
cf_fits = {}
print("=== HELD-OUT FEATURE DIAGNOSTICS ===")
for factual_weight in [1.0, 5.0, 20.0]:
    for ridge in [1.0, 10.0]:
        design = np.vstack(
            [np.sqrt(factual_weight) * cf_F0[cf_train],
             np.sqrt(5.0) * cf_Fother[cf_other_train]] +
            [dd[cf_train] for dd in cf_D])
        targets = np.vstack(
            [np.zeros((cf_train.sum(), 2)),
             np.zeros((cf_other_train.sum(), 2))] +
            [tt[cf_train] for tt in cf_targets])
        coef = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]),
                               design.T @ targets)
        cf_fits[(factual_weight, ridge)] = coef
        factual_rms = np.sqrt(np.mean((cf_F0[cf_test] @ coef) ** 2, axis=0))
        other_rms = np.sqrt(np.mean((cf_Fother[~cf_other_train] @ coef) ** 2, axis=0))
        metrics = []
        for drug_i in range(2):
            dd = np.concatenate([cf_D[2*drug_i][cf_test], cf_D[2*drug_i+1][cf_test]])
            yy = np.concatenate([cf_targets[2*drug_i][cf_test, drug_i],
                                 cf_targets[2*drug_i+1][cf_test, drug_i]])
            pp = dd @ coef[:, drug_i]
            active = yy != 0
            metrics.append((
                round(float(np.dot(pp[active], yy[active]) / np.dot(yy[active], yy[active])), 3),
                round(float(np.sqrt(np.mean((pp[active] - yy[active])**2))), 4)))
        print({"weight": factual_weight, "ridge": ridge,
               "factual_RMS": np.round(factual_rms, 4).tolist(),
               "nonaction_RMS": np.round(other_rms, 4).tolist(),
               "contrast_slope_RMSE": metrics})

print("=== CORE TENSORS ===")
print("\n".join(f"{name} {SHAPES[name]}" for name in NAMES
                if name.startswith("core.") or name.startswith("z_in.")))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 39: synthetic dose-contrast features\n"
            "Numerically verified drug bundles and fitted posterior-hidden features using recorded-dose zero constraints and synthetic x2/half dose contrasts. Diagnostics printed; no weights edited. Python runs: 39; forward passes: 3.\n")
```