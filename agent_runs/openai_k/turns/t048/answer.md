The innovation targets align exactly with the infusion bundles, so that is not the source of the centering error. I’ll now reconstruct the value readout and compare it with the original factual predictions. A close match would let me choose latent correction directions by their measured effects on both glucose and MAP, rather than by assumptions about the drugs.

```python
rr43_codes = {x["name"]: x["id"] for x in VOCAB["codes"]}
rr43_facts = train_factual0.sort_values(["stay_id", "event", "hour"]).copy()
rr43_group = rr43_facts.groupby(["stay_id", "event"], sort=False)
rr43_facts["last_value"] = rr43_group["observed"].shift()
rr43_facts["last_hour"] = rr43_group["hour"].shift()
rr43_facts = rr43_facts.iloc[np.linspace(0, len(rr43_facts)-1, 256).astype(int)].copy()
rr43_lookup = {key: i for i, key in enumerate(state_keys_audit)}
rr43_indices = np.array([rr43_lookup[(int(r.stay_id), float(r.hour))]
                         for r in rr43_facts.itertuples()])
rr43_ctx = contexts_audit[rr43_indices]
rr43_prior = mlp0(rr43_ctx, "z_prior")
print("Prior mean/log-SD quantiles:",
      [np.quantile(rr43_prior[:, sl].numpy(), [0.05, 0.5, 0.95]).tolist()
       for sl in [slice(0, 32), slice(32, 64)]])
rr43_view = rr43_ctx + lin0(rr43_prior[:, :32], "z_head")
rr43_cids = torch.tensor([rr43_codes[e] for e in rr43_facts.event], dtype=torch.long)
rr43_bins = []
rr43_zv = []
rr43_centres = []
rr43_scales = []
for rr43_row in rr43_facts.itertuples():
    rr43_spec = num_spec0[(rr43_row.event, "value")]
    rr43_has = pd.notna(rr43_row.last_value)
    rr43_bins.append(int(np.searchsorted(rr43_spec["fine_bin_edges"],
                                         rr43_row.last_value, side="left") + 1)
                     if rr43_has else 0)
    rr43_zv.append((rr43_row.last_value - rr43_spec["median"]) / rr43_spec["iqr"]
                   if rr43_has else 0.0)
    rr43_centres.append([0.0] + VOCAB["value_bin_centres"][rr43_row.event])
    rr43_scales.append(rr43_spec["iqr"])
rr43_bins = torch.tensor(rr43_bins, dtype=torch.long)
rr43_centres = torch.tensor(rr43_centres, dtype=torch.float32)
rr43_has = rr43_facts.last_value.notna().to_numpy().astype(np.float32)
rr43_age = np.log1p((rr43_facts.hour - rr43_facts.last_hour).fillna(0).to_numpy() * 3600) / 10
rr43_observed_mean = rr43_facts.predicted_mean.to_numpy()
rr43_scores = []
for rr43_type in [0, 1, 5]:
    rr43_q = norm0(rr43_view + W0["type_emb.weight"][rr43_type], "query_norm")
    for rr43_hascol in [None, 3, 4, 5, 6, 7]:
        rr43_numeric = torch.zeros(len(rr43_facts), 8)
        rr43_numeric[:, 0] = torch.tensor(rr43_has)
        rr43_numeric[:, 1] = torch.tensor(rr43_age, dtype=torch.float32)
        rr43_numeric[:, 2] = torch.tensor(rr43_zv, dtype=torch.float32)
        if rr43_hascol is not None:
            rr43_numeric[:, rr43_hascol] = torch.tensor(rr43_has)
        for rr43_status in [False, True]:
            rr43_wc = (rr43_q + W0["embed.code.weight"][rr43_cids]
                       + lin0(rr43_numeric, "last_numeric")
                       + W0["embed.bins.value.weight"][rr43_bins]
                       + W0["embed.bins.rate.weight"][0])
            if rr43_status:
                rr43_wc = rr43_wc + W0["last_status.weight"][0]
            rr43_logits = (mlp0(rr43_wc, "attr_heads.value_bin")
                           + W0["embed_transition.value_bin.weight"][rr43_cids*17 + rr43_bins])
            rr43_probs = torch.softmax(rr43_logits, dim=-1)
            rr43_pred = (rr43_probs * rr43_centres).sum(-1).numpy()
            rr43_error = rr43_pred - rr43_observed_mean
            rr43_score = float(np.sqrt(np.mean((rr43_error / np.array(rr43_scales))**2)))
            rr43_scores.append((rr43_score, rr43_type, rr43_hascol, rr43_status,
                                float(np.sqrt(np.mean(rr43_error**2)))))
print("\n=== VALUE-HEAD CONVENTION SEARCH ===")
print(sorted(rr43_scores)[:8])
print("\n=== DOSE-ROLLOUT OUTPUT SCHEMA ===")
rr43_rollout = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print(rr43_rollout.columns.tolist())
print(rr43_rollout.head(3).to_string(index=False))
print("\n=== DYNAMIC ESTIMATE STRUCTURE ===")
for rr43_key, rr43_obj in dynamic_ir.items():
    print(rr43_key, type(rr43_obj), str(rr43_obj)[:650])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 43: value-head reconstruction assay\n"
            "- Innovation alignment exactly matches the retained targets.\n"
            f"- Best value-head conventions (prior mean approximation): {sorted(rr43_scores)[:4]}\n"
            "- No tensors edited. Python runs: 43; forward passes: 4.\n")
```