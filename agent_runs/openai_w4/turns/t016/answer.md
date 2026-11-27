Prior averaging did not explain the mismatch. The largest errors are in glucose, pointing to the last-value input layout as a likely culprit. I’ll test those field positions systematically. A sharp reduction in reconstruction error would support that explanation; otherwise I’ll avoid relying on reconstructed heads for the repair.

```python
import itertools

torch.manual_seed(121)
sample_idx_layout = np.random.default_rng(121).choice(
    n_fact_repair, size=min(512, n_fact_repair), replace=False
)
si = torch.tensor(sample_idx_layout)
context_layout = context_base_repair + mlp_repair(gap_input_repair, "gap", "silu")
prior_layout = mlp_repair(context_layout, "z_prior", "silu")
q_layout = norm_repair(
    context_layout + linear_repair(prior_layout[:, :32], "z_head")
    + W_repair["type_emb.weight"][1], "query_norm"
)
base_layout = (
    q_layout + W_repair["embed.code.weight"][code_t_repair]
    + W_repair["embed.bins.value.weight"][bin_t_repair]
    + W_repair["last_status.weight"][0]
    + W_repair["last_numeric.bias"]
)
prebase_layout = linear_repair(base_layout, "attr_heads.value_bin.0")
last_to_hidden_layout = (
    W_repair["attr_heads.value_bin.0.weight"] @ W_repair["last_numeric.weight"]
).T
transition_layout = W_repair["embed_transition.value_bin.weight"][
    code_t_repair * 17 + bin_t_repair
]

layouts_repair = []
for zcol, agecol in itertools.permutations(range(8), 2):
    remaining = [j for j in range(8) if j not in (zcol, agecol)]
    for flag_columns in itertools.combinations(remaining, 2):
        layouts_repair.append((zcol, agecol, flag_columns))
    for flag_col in remaining:
        layouts_repair.append((zcol, agecol, (flag_col,)))

sample_has = torch.tensor(fact_has_repair[sample_idx_layout], dtype=torch.float32)
sample_age = torch.tensor(fact_age_repair[sample_idx_layout], dtype=torch.float32)
sample_z = torch.tensor(fact_last_z_repair[sample_idx_layout], dtype=torch.float32)
sample_centres = torch.from_numpy(fact_centres_repair[sample_idx_layout])
sample_obsbin = torch.tensor(fact_observed_bin_repair[sample_idx_layout])
sample_ref_mean = torch.tensor(reference_mean_repair[sample_idx_layout])
sample_ref_nll = torch.tensor(reference_nll_repair[sample_idx_layout])
layout_scores_repair = []

for start_layout in range(0, len(layouts_repair), 48):
    batch = layouts_repair[start_layout:start_layout + 48]
    vh = torch.stack([
        last_to_hidden_layout[list(flags)].sum(dim=0)
        for zcol, agecol, flags in batch
    ])
    vz = torch.stack([last_to_hidden_layout[zcol] for zcol, agecol, flags in batch])
    va = torch.stack([last_to_hidden_layout[agecol] for zcol, agecol, flags in batch])
    pre = (
        prebase_layout[si][None, :, :]
        + sample_has[None, :, None] * vh[:, None, :]
        + sample_z[None, :, None] * vz[:, None, :]
        + sample_age[None, :, None] * va[:, None, :]
    )
    for head_act in ("gelu", "silu"):
        logits = linear_repair(act_repair(pre, head_act), "attr_heads.value_bin.2")
        logits += transition_layout[si][None, :, :]
        prob = F.softmax(logits, dim=-1)
        pred = (prob * sample_centres[None, :, :]).sum(dim=-1)
        p_obs = prob.gather(
            2, sample_obsbin[None, :, None].expand(len(batch), -1, 1)
        ).squeeze(-1)
        nll = -torch.log(p_obs.clamp_min(1e-30))
        rms_mean = torch.sqrt(((pred - sample_ref_mean)**2).mean(dim=1)).numpy()
        rms_nll = torch.sqrt(((nll - sample_ref_nll)**2).mean(dim=1)).numpy()
        for k, (zcol, agecol, flags) in enumerate(batch):
            layout_scores_repair.append({
                "zcol": zcol, "agecol": agecol, "flags": flags,
                "head_act": head_act, "mean_RMSE": rms_mean[k], "nll_RMSE": rms_nll[k]
            })

layout_table_repair = pd.DataFrame(layout_scores_repair).sort_values("nll_RMSE")
print("=== BEST REGISTER LAYOUTS, 512-READING SEARCH ===")
print(layout_table_repair.head(10).round(6).to_string(index=False))

best_layout_row = layout_table_repair.iloc[0]
head_layout_repair = (
    int(best_layout_row.zcol), int(best_layout_row.agecol), tuple(best_layout_row.flags)
)
head_activation_repair = str(best_layout_row.head_act)
correct_last_fields_repair = np.zeros((n_fact_repair, 8), dtype=np.float32)
correct_last_fields_repair[:, head_layout_repair[0]] = fact_last_z_repair
correct_last_fields_repair[:, head_layout_repair[1]] = fact_age_repair
for col in head_layout_repair[2]:
    correct_last_fields_repair[:, col] = fact_has_repair
correct_last_t_repair = torch.from_numpy(correct_last_fields_repair)

def head_probs_repair(context, latent):
    q = norm_repair(
        context + linear_repair(latent, "z_head") + W_repair["type_emb.weight"][1],
        "query_norm"
    )
    with_code = (
        q + W_repair["embed.code.weight"][code_t_repair]
        + W_repair["embed.bins.value.weight"][bin_t_repair]
        + linear_repair(correct_last_t_repair, "last_numeric")
    )
    hidden = act_repair(linear_repair(with_code, "attr_heads.value_bin.0"),
                        head_activation_repair)
    logits = linear_repair(hidden, "attr_heads.value_bin.2") + transition_layout
    return F.softmax(logits, dim=-1).numpy()

layout_prob_mean = head_probs_repair(context_layout, prior_layout[:, :32])
layout_prob_mc = np.zeros_like(layout_prob_mean, dtype=np.float64)
for draw in range(32):
    latent = prior_layout[:, :32] + torch.exp(
        prior_layout[:, 32:].clamp(-6, 2)
    ) * torch.randn_like(prior_layout[:, :32])
    layout_prob_mc += head_probs_repair(context_layout, latent) / 32

layout_full_metrics_repair = []
for mode, prob in (("prior_mean", layout_prob_mean), ("32_draw_average", layout_prob_mc)):
    pred = (prob * fact_centres_repair).sum(axis=1)
    nll = -np.log(np.maximum(
        prob[np.arange(n_fact_repair), fact_observed_bin_repair], 1e-30
    ))
    layout_full_metrics_repair.append({
        "mode": mode,
        "mean_RMSE": np.sqrt(np.mean((pred-reference_mean_repair)**2)),
        "nll_RMSE": np.sqrt(np.mean((nll-reference_nll_repair)**2)),
        "mean_nll": nll.mean(),
    })
print("\nFull-data validation:")
print(pd.DataFrame(layout_full_metrics_repair).round(6).to_string(index=False))
print("Chosen last layout:", head_layout_repair, "head activation:", head_activation_repair)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 12\n"
        "Prior averaging did not fix reconstruction. Systematically tested "
        "last-numeric z, age and presence-flag positions on 512 readings, "
        "then validated best layout on all 4903 readings.\n"
        + layout_table_repair.head(5).round(6).to_string(index=False)
        + "\n" + pd.DataFrame(layout_full_metrics_repair).round(6).to_string(index=False)
        + "\nChosen layout: " + str(head_layout_repair)
        + "; head activation: " + head_activation_repair + "\nNo edits.\n"
    )
```