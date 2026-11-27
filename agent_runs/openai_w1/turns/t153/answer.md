The run stopped before any weight edit: the head helper passed an activation name to a function expecting a callable. This is an interface error, not evidence against the repair approach. I’ll fix that dispatch and retry, checking the reconstructed head against the GPU factual predictions before starting the probe.

```python
he_old_mlp = mlp
def mlp(x, prefix, kind):
    if isinstance(kind, str):
        activation = {
            "silu": torch.nn.functional.silu,
            "gelu": torch.nn.functional.gelu,
            "relu": torch.nn.functional.relu,
        }[kind]
    else:
        activation = kind
    return lin(activation(lin(x, prefix + ".0")), prefix + ".2")

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
he_rmse = float(np.sqrt(np.mean(
    (he_check.cpu_mean - he_check.predicted_mean)**2)))
he_nll_diff = float((he_check.cpu_nll - he_check.nll).mean())
print("paired stays/readings:", len(he_ids), len(he_mean0))
print("head-vs-forward mean RMSE:", he_rmse)
print("head-vs-forward mean NLL difference:", he_nll_diff)

he_kernels = {}
for he_key, he_obj in repair_lp.items():
    he_yr = np.array([x[0] for x in he_obj["y_response"]])
    he_dr = np.array([x[0] for x in he_obj["dose_response"]])
    he_kk = np.zeros(len(he_yr))
    for he_jj in range(len(he_yr)):
        he_kk[he_jj] = he_yr[he_jj] - sum(
            he_dr[he_ll] * he_kk[he_jj-he_ll]
            for he_ll in range(1, he_jj+1))
    he_kernels[he_key] = he_kk
    print("record kernel", he_key, np.round(he_kk, 4))
print("drug normalization:", {
    str(k): {f: v.get(f) for f in ("median", "iqr")}
    for k, v in ce_meta.items() if k[0] in repair_drugs
})

he_probe_scale = 1.5
he_numeric_original = torch.as_tensor(
    load_original("embed.numeric.weight"), dtype=torch.float32).clone()
if he_rmse < 3.0 and abs(he_nll_diff) < 0.05:
    he_numeric_probe = he_numeric_original.clone()
    he_numeric_probe[:, [2, 4]] *= he_probe_scale
    save_edit("embed.numeric.weight", he_numeric_probe)
    print("FORWARD STAYS:", json.dumps(he_ids))
else:
    print("Probe withheld: factual-head reconstruction needs correction.")
print("edits:", edited())

with open("/work/notes.md", "a") as he_f:
    he_f.write(
        "\n## Runs 139–140\n"
        "Run 139 failed before editing because mlp activation dispatch expected "
        "a callable. Run 140 repaired string/callable dispatch and checked "
        f"head reconstruction: mean RMSE={he_rmse:.5g}, "
        f"mean NLL difference={he_nll_diff:.5g}. "
        "If embed.numeric.weight is edited, this is ONLY the temporary "
        "1.5x centered dose/rate sensitivity probe; restore after states export. "
        "Counts: 140 Python runs, 7 forwards.\n"
    )
```