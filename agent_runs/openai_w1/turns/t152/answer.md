I’ll test a more direct route: temporarily perturb the drug-number embedding, export the resulting states, then restore it. Paired states can reveal dose-sensitive head features without requiring CPU replay. This route is worth keeping only if an algebraic head edit can strengthen dose responses with little factual loss.

```python
# Prepare paired-state diagnostics. This is a temporary probe, not a final repair.
he_train_ids = np.array(sorted({
    int(s) for s in ce_states["stay_id"] if str(int(s)).startswith("101")
}))
he_ids = he_train_ids[::2][:100].tolist()
assert len(he_ids) >= 50

he_mask = np.isin(ce_states["stay_id"], he_ids)
he_s0 = {k: np.asarray(v)[he_mask].copy() for k, v in ce_states.items()}
he_records = train[train.stay_id.isin(he_ids)].copy()
he_base = ce_make_heads(he_records, he_s0)

he_phi0 = he_base["phi"].detach().cpu().numpy().astype(np.float64)
he_p0 = he_base["probs"].detach().cpu().numpy()
he_centres = he_base["centres"].detach().cpu().numpy()
he_mean0 = (he_p0 * he_centres).sum(1)
he_bins = np.asarray(he_base["truebin"], dtype=int)
he_nll0 = -np.log(np.maximum(he_p0[np.arange(len(he_bins)), he_bins], 1e-30))
he_check = he_base["rows"][["stay_id", "hour", "event"]].copy()
he_check["cpu_mean"] = he_mean0
he_check["cpu_nll"] = he_nll0
he_check = he_check.merge(
    ce_factual[["stay_id", "hour", "event", "predicted_mean", "nll"]],
    on=["stay_id", "hour", "event"], how="inner"
)
print("paired stays:", len(he_ids), "readings:", len(he_mean0))
print("head-vs-forward mean RMSE:",
      np.sqrt(np.mean((he_check.cpu_mean - he_check.predicted_mean)**2)))
print("head-vs-forward mean NLL difference:",
      (he_check.cpu_nll - he_check.nll).mean())

he_kernels = {}
for he_key, he_obj in repair_lp.items():
    he_yr = np.array([x[0] for x in he_obj["y_response"]])
    he_dr = np.array([x[0] for x in he_obj["dose_response"]])
    he_kk = np.zeros(len(he_yr))
    for he_jj in range(len(he_yr)):
        he_kk[he_jj] = he_yr[he_jj] - sum(
            he_dr[he_ll] * he_kk[he_jj-he_ll]
            for he_ll in range(1, he_jj+1)
        )
    he_kernels[he_key] = he_kk
    print("record kernel", he_key, np.round(he_kk, 4))
print("drug normalization:",
      {k: v for k, v in ce_meta.items() if k[0] in repair_drugs})

he_probe_scale = 1.5
he_numeric_original = torch.as_tensor(
    load_original("embed.numeric.weight"), dtype=torch.float32
).clone()
assert tuple(he_numeric_original.shape) == (192, 6)
he_numeric_probe = he_numeric_original.clone()
he_numeric_probe[:, [2, 4]] *= he_probe_scale
save_edit("embed.numeric.weight", he_numeric_probe)

print("FORWARD STAYS:", json.dumps(he_ids))
print("temporary edits:", edited())

with open("/work/notes.md", "a") as he_f:
    he_f.write(
        "\n## Run 139: paired-state sensitivity probe\n"
        "Prepared original head features on 100 or fewer exported training stays. "
        "TEMPORARY edit: embed.numeric.weight columns 2 and 4 scaled by 1.5 "
        "(centered dose/rate numbers only). Restore this tensor immediately "
        "after the next states export; it is not a proposed final repair. "
        "Use paired head-feature changes to test a factual-constrained, "
        "gradient-free value-head edit. Counts: 139 Python runs, 7 forwards.\n"
    )
```