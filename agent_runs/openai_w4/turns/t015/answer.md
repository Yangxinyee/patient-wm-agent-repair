The head reconstruction is still materially wrong, so I won’t use it to compute an edit yet. I’ll check prior sampling, time-gap scaling and last-register fields. A close match to the existing factual predictions would validate the calculation; persistent disagreement would mean another implementation detail is missing.

```python
def act_repair(x, kind):
    if kind == "gelu":
        return F.gelu(x)
    if kind == "silu":
        return F.silu(x)
    if kind == "tanh":
        return torch.tanh(x)
    return F.relu(x)

core_trials_repair = []
core_trial_cache_repair = {}
last_fields_trials_repair = {}
for flag_col in range(3, 8):
    fields = np.zeros((n_fact_repair, 8), dtype=np.float32)
    fields[:, 0] = fact_has_repair
    fields[:, 1] = fact_age_repair
    fields[:, 2] = fact_last_z_repair
    fields[:, flag_col] = fact_has_repair
    last_fields_trials_repair[flag_col] = torch.from_numpy(fields)

for core_act in ("silu", "gelu"):
    for gap_factor in (1.0, 10.0):
        context = context_base_repair + mlp_repair(
            gap_input_repair * gap_factor, "gap", core_act
        )
        prior = mlp_repair(context, "z_prior", core_act)
        for latent_mode in ("mean", "zero", "previous_posterior"):
            latent = {
                "mean": prior[:, :32],
                "zero": torch.zeros_like(prior[:, :32]),
                "previous_posterior": fact_state_repair[:, 960:992],
            }[latent_mode]
            view = context + linear_repair(latent, "z_head")
            q = norm_repair(
                view + W_repair["type_emb.weight"][1], "query_norm"
            )
            base = (
                q + W_repair["embed.code.weight"][code_t_repair]
                + W_repair["embed.bins.value.weight"][bin_t_repair]
                + W_repair["embed.bins.rate.weight"][0]
            )
            for flag_col, fields in last_fields_trials_repair.items():
                for status_mode in ("zero", "seen"):
                    status = W_repair["last_status.weight"][0].expand(n_fact_repair, -1)
                    if status_mode == "seen":
                        has_t = torch.tensor(fact_has_repair, dtype=torch.long)
                        status = W_repair["last_status.weight"][has_t]
                    with_code = base + status + linear_repair(fields, "last_numeric")
                    prehidden = linear_repair(with_code, "attr_heads.value_bin.0")
                    for head_act in ("silu", "gelu"):
                        hidden = act_repair(prehidden, head_act)
                        logits = linear_repair(hidden, "attr_heads.value_bin.2")
                        logits += W_repair["embed_transition.value_bin.weight"][
                            code_t_repair * 17 + bin_t_repair
                        ]
                        prob = F.softmax(logits, dim=-1).numpy()
                        pred = (prob * fact_centres_repair).sum(axis=1)
                        nll = -np.log(np.maximum(
                            prob[np.arange(n_fact_repair), fact_observed_bin_repair], 1e-30
                        ))
                        key = (core_act, gap_factor, latent_mode,
                               flag_col, status_mode, head_act)
                        core_trials_repair.append({
                            "core_act": core_act, "gap_factor": gap_factor,
                            "latent": latent_mode, "value_flag_col": flag_col,
                            "status": status_mode, "head_act": head_act,
                            "mean_RMSE": np.sqrt(np.mean((pred-reference_mean_repair)**2)),
                            "nll_RMSE": np.sqrt(np.mean((nll-reference_nll_repair)**2)),
                            "mean_nll": nll.mean(),
                        })
                        core_trial_cache_repair[key] = (pred, nll)

core_trial_table_repair = pd.DataFrame(core_trials_repair).sort_values("nll_RMSE")
print("=== DETERMINISTIC RECONSTRUCTION CHECKS ===")
print(core_trial_table_repair.head(12).round(5).to_string(index=False))

best_trial_repair = core_trial_table_repair.iloc[0]
best_trial_key_repair = (
    best_trial_repair.core_act, float(best_trial_repair.gap_factor),
    best_trial_repair.latent, int(best_trial_repair.value_flag_col),
    best_trial_repair.status, best_trial_repair.head_act
)
best_pred_repair, best_nll_repair = core_trial_cache_repair[best_trial_key_repair]
check_detail_repair = baseline_fact.copy()
check_detail_repair["reconstructed"] = best_pred_repair
check_detail_repair["mean_error"] = best_pred_repair - reference_mean_repair
print("\nFirst predictions:\n", check_detail_repair.head(8).to_string(index=False))
print("\nMean error by code:\n",
      check_detail_repair.groupby("event").mean_error.agg(["mean", "std"]).to_string())

core_act = str(best_trial_repair.core_act)
context = context_base_repair + mlp_repair(
    gap_input_repair * float(best_trial_repair.gap_factor), "gap", core_act
)
prior = mlp_repair(context, "z_prior", core_act)
print("\nPrior log-sd percentiles:",
      np.percentile(prior[:, 32:].numpy(), [0, 10, 50, 90, 100]))
print("Last-status row norms:",
      torch.linalg.vector_norm(W_repair["last_status.weight"], dim=1).tolist())
for prefix in ("encode.norm", "encode_action.norm", "z_in", "z_head"):
    for suffix in ("weight", "bias"):
        name = prefix + "." + suffix
        if name in W_repair:
            print(name, "norm", float(torch.linalg.vector_norm(W_repair[name])),
                  "mean", float(W_repair[name].mean()))

# Test averaged prior probabilities, not probabilities at the mean latent.
torch.manual_seed(771)
mc_prob_repair = np.zeros((n_fact_repair, 17), dtype=np.float64)
fields = last_fields_trials_repair[int(best_trial_repair.value_flag_col)]
has_t = torch.tensor(fact_has_repair, dtype=torch.long)
status = W_repair["last_status.weight"][
    has_t if best_trial_repair.status == "seen" else torch.zeros_like(has_t)
]
for draw in range(32):
    latent = prior[:, :32] + torch.exp(prior[:, 32:].clamp(-6, 2)) * torch.randn_like(prior[:, :32])
    q = norm_repair(
        context + linear_repair(latent, "z_head") + W_repair["type_emb.weight"][1],
        "query_norm"
    )
    with_code = (
        q + W_repair["embed.code.weight"][code_t_repair]
        + W_repair["embed.bins.value.weight"][bin_t_repair]
        + status + linear_repair(fields, "last_numeric")
    )
    hidden = act_repair(linear_repair(with_code, "attr_heads.value_bin.0"),
                        str(best_trial_repair.head_act))
    logits = linear_repair(hidden, "attr_heads.value_bin.2")
    logits += W_repair["embed_transition.value_bin.weight"][code_t_repair*17+bin_t_repair]
    mc_prob_repair += F.softmax(logits, dim=-1).numpy() / 32
mc_mean_repair = (mc_prob_repair * fact_centres_repair).sum(axis=1)
mc_nll_repair = -np.log(np.maximum(
    mc_prob_repair[np.arange(n_fact_repair), fact_observed_bin_repair], 1e-30
))
mc_metrics_repair = {
    "mean_RMSE": float(np.sqrt(np.mean((mc_mean_repair-reference_mean_repair)**2))),
    "nll_RMSE": float(np.sqrt(np.mean((mc_nll_repair-reference_nll_repair)**2))),
    "mean_nll": float(mc_nll_repair.mean()),
}
print("\n32-draw prior-average check:", mc_metrics_repair)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 11\n"
        "Run 10 head reconstruction failed (best mean RMSE 7.98, NLL RMSE 0.579); "
        "not trusted for editing.\n"
        "Checked latent source, gap scale, register flags/status and mixed activations.\n"
        + core_trial_table_repair.head(6).round(5).to_string(index=False)
        + "\n32-draw prior-average metrics: " + str(mc_metrics_repair)
        + "\nNo edits.\n"
    )
```