GELU throughout the gap, prior and value head gives a much closer match; measurement query type 1 is also clearly correct. I’ll repeat the register search with those computations fixed, then compare the best reconstruction against prior-sampling variability. That should establish whether the remaining discrepancy is small enough for controlled readout edits.

```python
import itertools
import numpy as np
import pandas as pd
import torch

ga, pa, gf, hact = joint_best_key
sample_static_wc = joint_best["q"] + base_layout[si] - q_layout[si]
sample_static_pre = linear_repair(sample_static_wc, "attr_heads.value_bin.0")
register_to_pre = load("last_numeric.weight").T @ load("attr_heads.value_bin.0.weight").T

packing_rows = []
for zcol_test in range(8):
    for agecol_test in range(8):
        if agecol_test == zcol_test:
            continue
        available_flags = [j for j in range(8) if j not in (zcol_test, agecol_test)]
        flag_sets = [
            fs for size in range(4)
            for fs in itertools.combinations(available_flags, size)
        ]
        flag_matrix = torch.zeros((len(flag_sets), 8))
        for j, fs in enumerate(flag_sets):
            for fc in fs:
                flag_matrix[j, fc] = 1.0
        flag_pre = flag_matrix @ register_to_pre
        pre_grid = (
            sample_static_pre[None]
            + sample_z[None, :, None] * register_to_pre[zcol_test][None, None]
            + sample_age[None, :, None] * register_to_pre[agecol_test][None, None]
            + sample_has[None, :, None] * flag_pre[:, None, :]
        )
        logits_grid = (
            linear_repair(act_repair(pre_grid, hact), "attr_heads.value_bin.2")
            + transition_layout[si][None]
        )
        probs_grid = torch.softmax(logits_grid, dim=-1)
        means_grid = (probs_grid * sample_centres[None]).sum(-1)
        nlls_grid = -torch.log(probs_grid.gather(
            2, sample_obsbin[None, :, None].expand(len(flag_sets), -1, -1)
        ).squeeze(-1).clamp_min(1e-12))
        mean_rms_grid = (
            (means_grid-sample_ref_mean[None]).square().mean(1).sqrt().numpy()
        )
        nll_rms_grid = (
            (nlls_grid-sample_ref_nll[None]).square().mean(1).sqrt().numpy()
        )
        for j, fs in enumerate(flag_sets):
            packing_rows.append((
                zcol_test, agecol_test, fs,
                float(mean_rms_grid[j]), float(nll_rms_grid[j]),
                float(nlls_grid[j].mean())
            ))

packing_table_repair = pd.DataFrame(packing_rows, columns=[
    "zcol", "agecol", "flags", "mean_RMSE", "nll_RMSE", "mean_nll"
]).sort_values("nll_RMSE")
print("=== REGISTER SEARCH WITH CORRECTED VIEW ===")
print(packing_table_repair.head(15).round(6).to_string(index=False))

packing_best = packing_table_repair.iloc[0]
head_cfg_repair = {
    "gap_activation": ga, "prior_activation": pa,
    "head_gap_factor": gf, "head_activation": hact, "type": 1,
    "zcol": int(packing_best["zcol"]),
    "agecol": int(packing_best["agecol"]),
    "flags": tuple(packing_best["flags"]),
}
print("\nProvisional head configuration:", head_cfg_repair)

# Full factual reconstruction with the prior mean.
ctx_full_repair = context_base_repair + mlp_repair(gap_input_repair, "gap", ga)
prior_full_repair = mlp_repair(ctx_full_repair, "z_prior", pa)
view_full_repair = (
    context_base_repair
    + mlp_repair(gap_input_repair * gf, "gap", ga)
    + linear_repair(prior_full_repair[:, :32], "z_head")
)
q_full_repair = norm_repair(
    view_full_repair + load("type_emb.weight")[1], "query_norm"
)
register_fields_repair = torch.zeros((len(baseline_fact), 8))
register_fields_repair[:, head_cfg_repair["zcol"]] = torch.as_tensor(
    fact_last_z_repair, dtype=torch.float32
)
register_fields_repair[:, head_cfg_repair["agecol"]] = torch.as_tensor(
    fact_age_repair, dtype=torch.float32
)
for fc in head_cfg_repair["flags"]:
    register_fields_repair[:, fc] = torch.as_tensor(
        fact_has_repair, dtype=torch.float32
    )
wc_full_repair = (
    q_full_repair + base_layout - q_layout
    + register_fields_repair @ load("last_numeric.weight").T
)
hidden_full_repair = act_repair(
    linear_repair(wc_full_repair, "attr_heads.value_bin.0"), hact
)
prob_full_repair = torch.softmax(
    linear_repair(hidden_full_repair, "attr_heads.value_bin.2")
    + transition_layout, dim=-1
)
mean_full_repair = (prob_full_repair * torch.as_tensor(
    fact_centres_repair, dtype=torch.float32
)).sum(-1)
nll_full_repair = -torch.log(prob_full_repair.gather(
    1, torch.as_tensor(fact_observed_bin_repair, dtype=torch.long)[:, None]
).squeeze(1).clamp_min(1e-12))
full_check_repair = baseline_fact.copy()
full_check_repair["mean_error"] = mean_full_repair.numpy() - reference_mean_repair
full_check_repair["nll_error"] = nll_full_repair.numpy() - reference_nll_repair
print("\n=== FULL FACTUAL CHECK ===")
print("mean RMSE:", float(np.sqrt(np.mean(full_check_repair.mean_error**2))))
print("NLL RMSE:", float(np.sqrt(np.mean(full_check_repair.nll_error**2))))
print(full_check_repair.groupby("event")[["mean_error", "nll_error"]]
      .agg(["mean", "std"]).round(5).to_string())

# Monte Carlo check on the independent 512-reading diagnostic subset.
gen_check = torch.Generator().manual_seed(418321)
mc_probs_best = []
p_mu_check = prior_full_repair[si, :32]
p_sd_check = prior_full_repair[si, 32:].clamp(-10, 5).exp()
static_wc_check = wc_full_repair[si] - q_full_repair[si]
for draw in range(32):
    zs = p_mu_check + p_sd_check * torch.randn(p_mu_check.shape, generator=gen_check)
    qs = norm_repair(
        context_base_repair[si]
        + mlp_repair(gap_input_repair[si] * gf, "gap", ga)
        + linear_repair(zs, "z_head") + load("type_emb.weight")[1],
        "query_norm"
    )
    hs = act_repair(
        linear_repair(qs + static_wc_check, "attr_heads.value_bin.0"), hact
    )
    mc_probs_best.append(torch.softmax(
        linear_repair(hs, "attr_heads.value_bin.2") + transition_layout[si], -1
    ))
mc_probs_best = torch.stack(mc_probs_best)
groups_best4 = mc_probs_best.reshape(8, 4, len(si), 17).mean(1)
means_best4 = (groups_best4 * sample_centres[None]).sum(-1)
nlls_best4 = -torch.log(groups_best4.gather(
    2, sample_obsbin[None, :, None].expand(8, -1, -1)
).squeeze(-1).clamp_min(1e-12))
print("\n=== CORRECTED PRIOR MONTE CARLO CHECK ===")
print("32-draw reference metrics:", sample_metrics_repair(mc_probs_best.mean(0)))
print("four-draw mean variability:",
      float(means_best4.std(0).square().mean().sqrt()))
print("four-draw NLL variability:",
      float(nlls_best4.std(0).square().mean().sqrt()))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 18\n"
        "Joint search favoured GELU for gap/prior/value head, gap factor 1, "
        "and measurement query type 1. Re-searched register packing with this "
        "view, reconstructed all 4,903 readings and assessed latent noise. "
        "Provisional head configuration: " + repr(head_cfg_repair)
        + ". No weights edited.\n"
    )
```