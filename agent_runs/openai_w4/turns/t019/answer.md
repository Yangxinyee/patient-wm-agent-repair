The tensor metadata reveals eight register inputs, while the architecture describes only four conceptually. That mismatch could explain the remaining error. I’ll reproduce the best candidate using explicit column indexing and inspect its inputs and errors. Consistent errors by event or time would indicate a missing deterministic field, rather than just random latent draws.

```python
import numpy as np
import pandas as pd
import torch

print("=== VIEW AND REGISTER TENSORS ===")
for name in NAMES:
    if (
        name.startswith("core.")
        and not any(s in name for s in ("prelude", "blocks", "coda"))
    ) or name.startswith(("last_", "query_", "type_", "embed.numeric")):
        print(name, SHAPES[name])
print("numeric_note:", VOCAB.get("numeric_note"))
print("last_numeric column norms:",
      load("last_numeric.weight").norm(dim=0).numpy().round(4))

print("\n=== PREVIOUS BEST CANDIDATES ===")
print(best_layout_row.to_string())
print(best_trial_repair.to_string())
print("check_detail columns:", list(check_detail_repair.columns))
print(check_detail_repair.head(5).to_string(index=False))

print("\n=== INPUT ALIGNMENT ===")
alignment_probe = baseline_fact.head(12).copy()
alignment_probe["prev_state_index"] = fact_prev_repair[:12]
alignment_probe["dt"] = fact_dt_repair[:12]
alignment_probe["has_last"] = fact_has_repair[:12]
alignment_probe["age"] = fact_age_repair[:12]
alignment_probe["last_z"] = fact_last_z_repair[:12]
alignment_probe["last_bin"] = fact_bin_repair[:12]
alignment_probe["state_hour"] = state_meta["hour"][fact_prev_repair[:12]]
print(alignment_probe.to_string(index=False))

print("\n=== SEARCH-INPUT CONVENTIONS ===")
idx_probe = np.asarray(sample_idx_layout[:6], dtype=int)
print("sample indices:", idx_probe)
for key in ("sample_has", "sample_age", "sample_z"):
    val = globals()[key]
    print(key, val[:6].detach().cpu().numpy())
print("raw has:", fact_has_repair[idx_probe])
print("raw age:", fact_age_repair[idx_probe])
print("raw z:", fact_last_z_repair[idx_probe])
print("log1p(age)/10:", np.log1p(fact_age_repair[idx_probe]) / 10)

# Explicit indexing avoids pandas' .flags attribute.
zc = int(best_layout_row["zcol"])
ac = int(best_layout_row["agecol"])
flag_cols = tuple(best_layout_row["flags"])
ha = str(best_layout_row["head_act"])
pre_best_sample = (
    prebase_layout[si]
    + sample_z[:, None] * last_to_hidden_layout[zc]
    + sample_age[:, None] * last_to_hidden_layout[ac]
)
for fc in flag_cols:
    pre_best_sample += sample_has[:, None] * last_to_hidden_layout[int(fc)]
logits_best_sample = (
    linear_repair(act_repair(pre_best_sample, ha), "attr_heads.value_bin.2")
    + transition_layout[si]
)
prob_best_sample = torch.softmax(logits_best_sample, dim=-1)
mean_best_sample = (prob_best_sample * sample_centres).sum(-1)
nll_best_sample = -torch.log(
    prob_best_sample.gather(1, sample_obsbin[:, None]).squeeze(1).clamp_min(1e-12)
)
sample_check = baseline_fact.iloc[np.asarray(sample_idx_layout)].copy()
sample_check["mean_error"] = (
    mean_best_sample.detach().numpy() - sample_ref_mean.detach().numpy()
)
sample_check["nll_error"] = (
    nll_best_sample.detach().numpy() - sample_ref_nll.detach().numpy()
)
print("\n=== REPRODUCED BEST REGISTER CANDIDATE ===")
print("mean RMSE:", float(np.sqrt(np.mean(sample_check.mean_error**2))))
print("NLL RMSE:", float(np.sqrt(np.mean(sample_check.nll_error**2))))
print(sample_check.groupby("event")[["mean_error", "nll_error"]]
      .agg(["mean", "std"]).round(4).to_string())
print("\nLargest NLL discrepancies:")
print(sample_check.iloc[np.argsort(np.abs(sample_check.nll_error.to_numpy()))[-8:]]
      .to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 15\n"
        "Investigated eight-dimensional last_numeric packing (architecture's "
        "description is abbreviated), reproduced the best register candidate "
        "using explicit column indexing, and examined alignment and residuals "
        "by event. No weights edited.\n"
    )
```