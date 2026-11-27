The latent pathway is almost insensitive to dose bundles: producing even a one-unit reading change would require an enormous perturbation. I won’t use that edit. Instead, I’ll test whether the state-update projection can carry a dose correction that is small on factual records. Held-out dose contrasts must be clearly larger than factual leakage for this alternative to be credible.

```python
import numpy as np
import torch
import torch.nn.functional as F

# Reconstruct prelude features, using the GPU posterior rather than the
# imperfect observation-posterior reconstruction.
n46 = len(state_df)
state_array46 = tt45(core_state)
meta46 = state_df.set_index("state_row").reindex(np.arange(n46))
ordered46 = state_df.sort_values(["stay_id", "hour"]).copy()
ordered46["previous_row46"] = ordered46.groupby("stay_id").state_row.shift()
ordered46["previous_hour46"] = ordered46.groupby("stay_id").hour.shift()
ordered46 = ordered46.set_index("state_row").reindex(np.arange(n46))
valid46 = ordered46.previous_row46.notna().to_numpy()
prev46 = ordered46.previous_row46.fillna(0).to_numpy(dtype=int)
seconds46 = ((ordered46.hour - ordered46.previous_hour46).fillna(0).to_numpy()*3600)
hflow46 = state_array46[torch.tensor(prev46), 192:960].reshape(-1, 4, 192).clone()
hflow46[torch.tensor(~valid46)] = 0
hflow46 *= torch.exp(
    -torch.tensor(seconds46, dtype=torch.float32)[:, None, None] *
    torch.exp(W0["core.flow.a"])[None, None, :]
)

def prelude46(h, e, u, z, batch=256):
    pieces = []
    for start in range(0, len(h), batch):
        stop = min(start + batch, len(h))
        tokens = torch.cat([
            h[start:stop] + W0["core.slots"][None],
            e[start:stop, None], u[start:stop, None],
            core_lin(z[start:stop], "z_in")[:, None]
        ], dim=1)
        result = core_layer(tokens, "core.prelude.0", 4)
        pieces.append(core_norm(result[:, :4], "core.prelude_norm"))
    return torch.cat(pieces)

prelude_all46 = prelude46(
    hflow46, tt45(recon_e), tt45(recon_u), state_array46[:, 960:992]
)
cf_idx46 = torch.tensor(np45(cf_rows).astype(int), dtype=torch.long)
prelude_cf46 = prelude_all46[cf_idx46]
print("GPU posterior vs recovered drug posterior RMS:",
      float((state_array46[cf_idx46, 960:992] - tt45(cf_Z0)).square().mean().sqrt()))

def edited_action46(drug, factor):
    bundles = []
    for row in cf_frame.itertuples(index=False):
        key = (int(row.stay_id), int(round(float(row.hour)*60)))
        atoms = []
        for record in cf_starts.get(key, []):
            if record.event == drug:
                record = record._replace(
                    rate=float(record.rate)*factor,
                    dose=float(record.dose)*factor
                )
            atoms.append(recon_record_atom(record, kind=0))
        bundles.append(atoms)
    return recon_sets(bundles, prefix="encode_action", empty_zero=True)

prelude_contrasts46 = {}
dose_contrasts46 = {}
for drug46, col46 in [("insulin", 0), ("norepinephrine", 1)]:
    for factor46 in [2.0, 0.5]:
        uedit46 = edited_action46(drug46, factor46)
        zedit46 = _post33(cf_rows, tt45(cf_E), uedit46)
        eedit46 = prelude46(hflow46[cf_idx46], tt45(cf_E), uedit46, zedit46)
        prelude_contrasts46[(drug46, factor46)] = np45(
            (eedit46 - prelude_cf46).mean(dim=1)
        )
        target46 = np.zeros((len(cf_rows), 2), dtype=np.float64)
        target46[:, col46] = np45(cf_doses)[:, col46] * (factor46 - 1)
        dose_contrasts46[(drug46, factor46)] = target46

features46 = np45(prelude_all46.mean(dim=1)).astype(np.float64)
scale46 = np.maximum(features46[valid46].std(axis=0), 1e-5)
factual_design46 = features46 / scale46
train_stays46 = np.unique(cf_frame.stay_id.to_numpy()[np45(cf_train).astype(bool)])
train_all46 = valid46 & meta46.stay_id.isin(train_stays46).to_numpy()
test_all46 = valid46 & ~meta46.stay_id.isin(train_stays46).to_numpy()
train_cf46 = np45(cf_train).astype(bool)
test_cf46 = np45(cf_test).astype(bool)
contrast_train46 = np.concatenate([
    delta[train_cf46]/scale46 for delta in prelude_contrasts46.values()
])
target_train46 = np.concatenate([
    delta[train_cf46] for delta in dose_contrasts46.values()
])
Gcontrast46 = contrast_train46.T @ contrast_train46
Rcontrast46 = contrast_train46.T @ target_train46
Gfactual46 = factual_design46[train_all46].T @ factual_design46[train_all46]
prelude_fits46 = {}
for anchor46 in [0.1, 1.0, 10.0]:
    for ridge46 in [0.001, 0.1]:
        coef46 = np.linalg.solve(
            Gcontrast46 + anchor46*Gfactual46 + ridge46*np.eye(192),
            Rcontrast46
        )
        prelude_fits46[(anchor46, ridge46)] = coef46 / scale46[:, None]
        leakage46 = factual_design46[test_all46] @ coef46
        diagnostics46 = []
        for col46, drug46 in enumerate(["insulin", "norepinephrine"]):
            dx46 = np.concatenate([
                (prelude_contrasts46[(drug46, f)][test_cf46]/scale46) @ coef46[:, col46]
                for f in [2.0, 0.5]
            ])
            dy46 = np.concatenate([
                dose_contrasts46[(drug46, f)][test_cf46, col46] for f in [2.0, 0.5]
            ])
            active46 = dy46 != 0
            slope46 = np.dot(dx46[active46], dy46[active46]) / np.dot(dy46[active46], dy46[active46])
            diagnostics46.append(round(float(slope46), 4))
        print("prelude fit", (anchor46, ridge46),
              "held-out contrast slopes", diagnostics46,
              "factual RMS", np.sqrt(np.mean(leakage46**2, axis=0)),
              "weight norms", np.linalg.norm(prelude_fits46[(anchor46, ridge46)], axis=0))

# Capture the original pre-coda state to measure a direct update sensitivity
# without repeating the whole recurrent block for every perturbation.
capture46 = {}
old_core_lin46 = core_lin
def capturing_core_lin46(x, prefix):
    if prefix == "core.c_out":
        capture46["hnew"] = x.clone()
    return old_core_lin46(x, prefix)
globals()["core_lin"] = capturing_core_lin46
try:
    hcheck46, ocheck46 = core_update_local(tt45(core_z), core_config45)
finally:
    globals()["core_lin"] = old_core_lin46
hnew46 = capture46["hnew"]

def coda46(hnew):
    y = core_layer(core_lin(hnew, "core.c_out"), "core.coda.0", 4)
    return y, core_norm(y.mean(dim=1), "core.out_norm")

hc46, oc46 = coda46(hnew46)
print("Captured coda verification:",
      float((hc46-hcheck46).abs().max()), float((oc46-ocheck46).abs().max()))
jac_bin46 = np.zeros((len(link_df45), 192))
delta46 = F.softplus(W0["core.delta_in"])
for dim46 in range(192):
    offset46 = torch.zeros_like(hnew46)
    offset46[:, :, dim46] = 2.0*delta46[dim46]
    hp46, op46 = coda46(hnew46 + offset46)
    hm46, om46 = coda46(hnew46 - offset46)
    jac_bin46[:, dim46] = np45(
        (next_means45(hp46, op46)-next_means45(hm46, om46))/4.0
    )
bin_directions46 = {}
for drug46, target46 in [
    ("insulin", np.array([-1.0, 0.0])),
    ("norepinephrine", np.array([0.0, 1.0]))
]:
    relevant46 = selected_meta45["has_"+drug46].to_numpy()[link_df45.core_index]
    J46 = np.stack([
        jac_bin46[relevant46 & (link_df45.event.to_numpy() == event)].mean(axis=0)
        for event in ["glucose_finger", "map_arterial"]
    ])
    direction46 = np.linalg.lstsq(J46, target46, rcond=1e-3)[0]
    bin_directions46[drug46] = direction46.astype(np.float32)
    print(drug46, "update direction norm", np.linalg.norm(direction46),
          "singular values", np.linalg.svd(J46, compute_uv=False),
          "response", J46 @ direction46)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 46: alternative prelude/update contrast probe\n"
            "Posterior-latent sensitivity was too weak for a credible edit. "
            "Tested factual-zero dose-contrast regressions on prelude features and direct update sensitivities. "
            "No weights edited. Python runs: 46; forward passes: 3.\n")
```
