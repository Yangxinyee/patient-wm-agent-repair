The saved states include the posterior latent mean, which gives us a direct check of the action encoder before reconstructing the whole recurrence. I’ll compare locally computed latent means across plausible unit mappings. A close match across drug bundles would confirm the conventions; a mismatch would mean local dose probes are not yet reliable.

```python
import torch.nn.functional as F
from itertools import product

for key in ("embed.kind.weight", "embed.ekind.weight", "embed.kq.weight",
            "embed.text.weight", "embed.categorical.unit.weight",
            "embed.categorical.rate_unit.weight",
            "embed.categorical.action.weight", "embed.categorical.status.weight",
            "embed.categorical.route.weight"):
    print(key, np.round(torch.linalg.vector_norm(W0[key], dim=1).numpy(), 4))
print("rate units:", train.loc[train.event.isin(["insulin", "norepinephrine"]),
                               ["event", "rate_unit", "dose_unit"]].drop_duplicates().to_dict("records"))

def atom_action_local(event, rate, kind=0, rate_unit=0, unit=0):
    rates = torch.as_tensor(np.atleast_1d(rate), dtype=torch.float32)
    num = torch.zeros(len(rates), 6)
    atom = W0["embed.code.weight"][code_id[event]].expand(len(rates), -1).clone()
    atom += W0["embed.kind.weight"][kind] + W0["embed.ekind.weight"][2]
    atom += W0["embed.categorical.rate_unit.weight"][rate_unit]
    atom += W0["embed.categorical.unit.weight"][unit]
    for attr, column, mult in (("dose", 2, 1.0 if event == "insulin" else 4.8),
                               ("rate", 4, 1.0)):
        meta = norm_meta[(event, attr)]
        values = rates * mult
        num[:, column] = torch.clamp((values-meta["median"])/meta["iqr"], -10, 10)
        num[:, column+1] = 1
        bins = 1 + np.searchsorted(meta["fine_bin_edges"], values.numpy(), side="left")
        atom += W0[f"embed.bins.{attr}.weight"][torch.as_tensor(bins)]
    return atom + lin0(num, "embed.numeric")

def set_encode_single_local(atoms, prefix):
    encoded = mlp0(atoms, prefix + ".mlp1", "gelu")
    size = torch.full((len(atoms), 1), np.log(2), dtype=torch.float32)
    return norm0(mlp0(encoded + lin0(size, prefix + ".size"),
                      prefix + ".mlp2", "gelu"), prefix + ".norm")

state4 = dict(np.load("/work/fwd_4/states_original.npz"))
drug_records_probe = train[train.stay_id.isin(np.unique(state4["stay_id"])) &
                           train.event.isin(["insulin", "norepinephrine"])].copy()
drug_records_probe["minute"] = np.rint(drug_records_probe.hour * 60).astype(int)
drug_lookup_probe = {
    (int(row.stay_id), int(row.minute), row.event): float(row.rate)
    for row in drug_records_probe.itertuples()
}
action_unit_trials = {}
action_probe_data = {}

for drug, flag in (("insulin", "has_insulin"),
                   ("norepinephrine", "has_norepinephrine")):
    rows = []
    rates = []
    for j in np.flatnonzero(state4[flag] & (state4["n_atoms"] == 1)):
        key = (int(state4["stay_id"][j]),
               int(round(state4["hour"][j] * 60)), drug)
        if j > 0 and state4["stay_id"][j-1] == state4["stay_id"][j] and key in drug_lookup_probe:
            rows.append(j)
            rates.append(drug_lookup_probe[key])
    rows = np.array(rows[:250])
    rates = np.array(rates[:250])
    if not len(rows):
        print(drug, "no single-start-atom probes")
        continue
    before = torch.from_numpy(state4["x"][rows-1])
    dt = torch.as_tensor(np.rint(
        (state4["hour"][rows]-state4["hour"][rows-1])*60)*60,
        dtype=torch.float32)[:, None]
    hf = before[:, 192:960].reshape(-1, 4, 192)
    hf = hf * torch.exp(-dt[:, :, None] * torch.exp(W0["core.flow.a"])[None, None, :])
    ctx = before[:, :192] + norm0(hf.mean(1), "core.out_norm")
    ctx += mlp0(torch.log1p(dt)/10, "gap", "gelu")
    target_z = torch.from_numpy(state4["x"][rows, 960:])
    trials = []
    for unit, rate_unit in product(range(6), range(4)):
        atoms = atom_action_local(drug, rates, rate_unit=rate_unit, unit=unit)
        action_u = set_encode_single_local(atoms, "encode_action")
        zp = mlp0(torch.cat([ctx, torch.zeros_like(ctx), action_u], dim=1),
                  "z_post", "gelu")[:, :32]
        rmse = float(torch.mean((zp-target_z)**2).sqrt())
        trials.append((rmse, unit, rate_unit))
    trials.sort()
    action_unit_trials[drug] = trials
    action_probe_data[drug] = dict(rows=rows, rates=rates, ctx=ctx, hf=hf, dt=dt)
    print(drug, "n", len(rows), "target_z RMS", float(target_z.square().mean().sqrt()),
          "best unit trials", trials[:6])
```
