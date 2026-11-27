The current helpers implement the documented formulas consistently, but still miss the forward predictions. I’ll check state timing and a bounded set of remaining ambiguities: time-flow placement, latent-output packing and the absent-dose embedding. A match near sampling noise would support one convention; unchanged error would point elsewhere.

```python
print("=== CHRONOLOGY CHECK ===")
state_times = STIME[hm.state_i.to_numpy(dtype=int)]
dt_from_archive = (hm.hour.to_numpy() - state_times) * 3600
print("maximum dt mismatch:",
      float(np.max(np.abs(dt_from_archive - hm.dt.to_numpy()))))
print("dt counts:", hm.dt.value_counts().head(8).to_dict())

print("\n=== NON-CORE STRUCTURAL TENSORS ===")
for name in NAMES:
    if (not name.startswith("core.")
        and not name.startswith(("attr_heads.", "embed_transition.", "embed.categorical."))
        and name not in W_original):
        print(name, SHAPES[name])

start_variants = len(variant_results)
q_default_input = view_default + wt("type_emb.weight")[1]

register_candidates = {
    "plus_absent_dose": reg_default + wt("embed.bins.dose.weight")[0],
    "without_absent_rate": reg_default - wt("embed.bins.rate.weight")[0],
    "plus_dose_without_rate": reg_default + wt("embed.bins.dose.weight")[0]
                                              - wt("embed.bins.rate.weight")[0],
    "without_status": reg_default - wt("last_status.weight")[0],
}
for label, candidate_reg in register_candidates.items():
    score_head_variant("check_" + label, q_default_input, reg=candidate_reg)

summaries = {
    "old_plus_flow": out + summary_check,
    "twice_flow": 2 * summary_check,
    "twice_old": 2 * out,
    "old_only": out,
    "flow_only": summary_check,
}
for prior_label, prior_base in summaries.items():
    prior_raw = mlp(prior_base + gap_default, "z_prior", "gelu")
    for pack_label, prior_mean in [
        ("split", prior_raw[:, :32]),
        ("interleaved", prior_raw[:, ::2]),
    ]:
        latent_view = TF.linear(prior_mean, wt("z_head.weight"), wt("z_head.bias"))
        for view_label, view_base in summaries.items():
            candidate_view = view_base + gap_default + latent_view
            score_head_variant(
                "check_" + prior_label + "_" + pack_label + "_" + view_label,
                candidate_view + wt("type_emb.weight")[1],
                reg=reg_default,
            )

check_table = pd.DataFrame(variant_results[start_variants:])
print("\n=== NEW CONVENTION CHECKS ===")
print(check_table.sort_values("scaled_mean_rmse").head(12).to_string(index=False))

print("\n=== PRIOR OUTPUT STATISTICS ===")
prior_raw = mlp(ctx_default, "z_prior", "gelu")
for label, part in [
    ("first_half", prior_raw[:, :32]),
    ("second_half", prior_raw[:, 32:]),
    ("even", prior_raw[:, ::2]),
    ("odd", prior_raw[:, 1::2]),
]:
    print(label, "mean", float(part.mean()), "sd", float(part.std()),
          "range", (float(part.min()), float(part.max())))

with open("/work/notes.md", "a") as notebook:
    notebook.write(
        "\n- Run 23: saved out equals layer-normalized mean h to 7.3e-8 RMSE. "
        "Current helpers use seconds-based flow, log1p(seconds)/10, "
        "interleaved last-number fields and measurement query type 1. "
        "These still give scaled-mean RMSE 0.114; no edits.\n"
    )
```