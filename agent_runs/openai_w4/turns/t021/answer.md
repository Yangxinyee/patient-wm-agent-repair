Four-draw randomness is smaller than the remaining reconstruction error, and ReLU fits better than the activations previously searched. Status terms do not help. I’ll now test the gap and prior computations jointly with the head activation, including the architecture’s different gap scaling for state context and prediction. A sharp improvement would identify the missing deterministic computation.

```python
import numpy as np
import pandas as pd
import torch

print("=== CONTEXT CHECK ===")
for activation in ("relu", "silu", "gelu"):
    candidate_context = context_base_repair + mlp_repair(
        gap_input_repair, "gap", activation
    )
    print(activation, "RMS versus retained context:",
          float((candidate_context-context_layout).square().mean().sqrt()))

# Keep the currently best register packing fixed while testing the view.
base_constant_sample = (
    base_layout[si] - q_layout[si]
    + sample_z[:, None] * Wlast[:, zc]
    + sample_age[:, None] * Wlast[:, ac]
)
for fc in flag_cols:
    base_constant_sample += sample_has[:, None] * Wlast[:, int(fc)]

joint_rows = []
joint_cache_repair = {}
for gap_activation in ("relu", "silu", "gelu"):
    context_candidate = context_base_repair[si] + mlp_repair(
        gap_input_repair[si], "gap", gap_activation
    )
    for prior_activation in ("relu", "silu", "gelu"):
        prior_candidate = mlp_repair(
            context_candidate, "z_prior", prior_activation
        )
        for head_gap_factor in (0.1, 1.0, 10.0):
            view_candidate = (
                context_base_repair[si]
                + mlp_repair(gap_input_repair[si] * head_gap_factor,
                             "gap", gap_activation)
                + linear_repair(prior_candidate[:, :32], "z_head")
            )
            q_candidate = norm_repair(
                view_candidate + load("type_emb.weight")[1], "query_norm"
            )
            wc_candidate = q_candidate + base_constant_sample
            pre_candidate = linear_repair(wc_candidate, "attr_heads.value_bin.0")
            for head_activation in ("relu", "silu", "gelu"):
                p_candidate = torch.softmax(
                    linear_repair(act_repair(pre_candidate, head_activation),
                                  "attr_heads.value_bin.2")
                    + transition_layout[si], dim=-1
                )
                metrics = sample_metrics_repair(p_candidate)
                key = (gap_activation, prior_activation, head_gap_factor, head_activation)
                joint_rows.append((*key, *metrics))
                joint_cache_repair[key] = {
                    "context": context_candidate,
                    "prior": prior_candidate,
                    "q": q_candidate,
                    "with_code": wc_candidate,
                    "prob": p_candidate,
                }

joint_table_repair = pd.DataFrame(joint_rows, columns=[
    "gap_activation", "prior_activation", "head_gap_factor", "head_activation",
    "mean_RMSE", "nll_RMSE", "mean_nll"
]).sort_values("nll_RMSE")
print("\n=== JOINT VIEW/HEAD SEARCH ===")
print(joint_table_repair.head(15).round(6).to_string(index=False))

joint_best_row = joint_table_repair.iloc[0]
joint_best_key = (
    joint_best_row["gap_activation"], joint_best_row["prior_activation"],
    float(joint_best_row["head_gap_factor"]), joint_best_row["head_activation"]
)
joint_best = joint_cache_repair[joint_best_key]
print("\nBest configuration:", joint_best_key)

# Check whether the code-query type agrees with WORLD measurement (type 1).
type_rows = []
fixed_view = (
    context_base_repair[si]
    + mlp_repair(gap_input_repair[si] * joint_best_key[2],
                 "gap", joint_best_key[0])
    + linear_repair(joint_best["prior"][:, :32], "z_head")
)
for type_index in range(load("type_emb.weight").shape[0]):
    q_type = norm_repair(fixed_view + load("type_emb.weight")[type_index],
                        "query_norm")
    p_type = torch.softmax(
        linear_repair(
            act_repair(linear_repair(q_type + base_constant_sample,
                                    "attr_heads.value_bin.0"),
                       joint_best_key[3]),
            "attr_heads.value_bin.2"
        ) + transition_layout[si], dim=-1
    )
    type_rows.append((type_index, *sample_metrics_repair(p_type)))
type_table_repair = pd.DataFrame(
    type_rows, columns=["type", "mean_RMSE", "nll_RMSE", "mean_nll"]
).sort_values("nll_RMSE")
print("\n=== QUERY TYPE CHECK ===")
print(type_table_repair.head(5).round(6).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 17\n"
        "Status corrections did not improve reconstruction. Four-draw prior "
        "variability (mean RMS 1.06; NLL RMS 0.123) was too small to explain "
        "the mismatch. ReLU outperformed SiLU/GELU with the current register. "
        "Jointly searched gap/prior/head activations, prediction-gap scaling "
        "and code-query type. Best provisional configuration: "
        + repr(joint_best_key)
        + ". No weights edited.\n"
    )
```