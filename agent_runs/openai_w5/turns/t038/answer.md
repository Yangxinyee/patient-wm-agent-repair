The insulin latent means now match almost exactly: the empty measurement set is encoded rather than replaced by zero, and the quality embedding is included. Norepinephrine still disagrees. I’ll check decimal-bin handling using the doses actually written in the records, since boundary rounding could explain that discrepancy.

```python
action_probe_exact_doses = {}
dose_lookup_probe = {
    (int(row.stay_id), int(row.minute), row.event): float(row.dose)
    for row in drug_records_probe.itertuples()
}

def action_atom_exact(event, rates, doses=None, kind=0, bin_mode="raw64"):
    raw_rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    raw_doses = (np.round(raw_rates * (1 if event == "insulin" else 4.8), 8)
                 if doses is None else np.asarray(doses, dtype=np.float64).reshape(-1))
    n = len(raw_rates)
    num = torch.zeros(n, 6)
    unit = 3 if event == "insulin" else 4
    rate_unit = 2 if event == "insulin" else 3
    atom = W0["embed.code.weight"][code_id[event]].expand(n, -1).clone()
    atom += W0["embed.kind.weight"][kind] + W0["embed.ekind.weight"][2]
    atom += W0["embed.kq.weight"][0]
    atom += W0["embed.categorical.unit.weight"][unit]
    atom += W0["embed.categorical.rate_unit.weight"][rate_unit]
    for attr, column, values in (("dose", 2, raw_doses), ("rate", 4, raw_rates)):
        meta = norm_meta[(event, attr)]
        num[:, column] = torch.as_tensor(
            np.clip((values-meta["median"])/meta["iqr"], -10, 10), dtype=torch.float32)
        num[:, column+1] = 1
        edges = np.asarray(meta["fine_bin_edges"])
        if bin_mode == "float32":
            bins = 1 + np.searchsorted(edges.astype(np.float32),
                                        values.astype(np.float32), side="left")
        elif bin_mode == "right64":
            bins = 1 + np.searchsorted(edges, values, side="right")
        elif bin_mode == "rounded_edges":
            bins = 1 + np.searchsorted(np.round(edges, 8), np.round(values, 8), side="left")
        else:
            bins = 1 + np.searchsorted(edges, values, side="left")
        atom += W0[f"embed.bins.{attr}.weight"][torch.as_tensor(bins)]
    return atom + lin0(num, "embed.numeric")

action_bin_trials = {}
for drug, probe in action_probe_data.items():
    doses = np.array([
        dose_lookup_probe[(int(state4["stay_id"][j]),
                           int(round(state4["hour"][j]*60)), drug)]
        for j in probe["rows"]
    ])
    action_probe_exact_doses[drug] = doses
    n = len(doses)
    e = norm0(mlp0(lin0(torch.zeros(n, 1), "encode.size"),
                   "encode.mlp2", "gelu"), "encode.norm")
    target_z = torch.from_numpy(state4["x"][probe["rows"], 960:])
    results = []
    for bin_mode in ("raw64", "float32", "right64", "rounded_edges"):
        atoms = action_atom_exact(drug, probe["rates"], doses, bin_mode=bin_mode)
        action_u = set_encode_single_local(atoms, "encode_action")
        zp = mlp0(torch.cat([probe["ctx"], e, action_u], dim=1), "z_post", "gelu")[:, :32]
        row_error = (zp-target_z).square().mean(1).sqrt().numpy()
        results.append((float(np.mean(row_error**2)**0.5),
                        bin_mode, float(np.median(row_error)), float(np.max(row_error))))
    action_bin_trials[drug] = sorted(results)
    print(drug, "dose/rate ratio range",
          np.min(doses/probe["rates"]), np.max(doses/probe["rates"]),
          "bin trials:", action_bin_trials[drug])
```
