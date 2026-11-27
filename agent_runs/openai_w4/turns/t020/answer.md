The best candidate is reproducible, but its errors differ across glucose and MAP, so I still suspect a deterministic mismatch. The age input is already log-scaled and must not be transformed again. I’ll test status terms and quantify the variability from four prior draws; reconstruction errors consistent with that variability would support using this head for edit screening.

```python
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)
print("=== LATENT/GAP NAMES ===")
for name in NAMES:
    if any(s in name.lower() for s in ("prior", "posterior", "z_head", "z_to", "latent", "gap")):
        print(name, SHAPES[name])
print("retained dicts:",
      [(k, len(v), list(v)[:4]) for k, v in list(globals().items())
       if isinstance(v, dict) and not k.startswith("_") and k not in ("SHAPES", "VOCAB")])

W0 = load("attr_heads.value_bin.0.weight")
Wlast = load("last_numeric.weight")
status_weights = load("last_status.weight")
print("\n=== BASE DECOMPOSITION CHECK ===")
raw_base_probe = (
    q_layout
    + load("embed.code.weight")[code_t_repair]
    + load("embed.bins.value.weight")[bin_t_repair]
    + load("embed.bins.rate.weight")[0]
    + load("last_numeric.bias")
)
for label, candidate in [
    ("no status", raw_base_probe),
    ("status 0", raw_base_probe + status_weights[0]),
    ("status 1", raw_base_probe + status_weights[1]),
]:
    print(label, "base RMS difference:",
          float((base_layout - candidate).square().mean().sqrt()))
print("prebase versus linear(base):",
      float((prebase_layout - linear_repair(base_layout, "attr_heads.value_bin.0"))
            .square().mean().sqrt()))

def sample_metrics_repair(p):
    m = (p * sample_centres).sum(-1)
    n = -torch.log(p.gather(1, sample_obsbin[:, None]).squeeze(1).clamp_min(1e-12))
    return (
        float((m - sample_ref_mean).square().mean().sqrt()),
        float((n - sample_ref_nll).square().mean().sqrt()),
        float(n.mean()),
    )

# Status corrections are relative to the currently reproduced candidate.
status_deltas = {
    "none": torch.zeros_like(pre_best_sample),
    "add_0": status_weights[0].expand_as(pre_best_sample),
    "add_1": status_weights[1].expand_as(pre_best_sample),
    "has_0": sample_has[:, None] * status_weights[0],
    "has_1": sample_has[:, None] * status_weights[1],
    "absent_0": (1-sample_has[:, None]) * status_weights[0],
    "absent_1": (1-sample_has[:, None]) * status_weights[1],
    "subtract_0": -status_weights[0].expand_as(pre_best_sample),
    "subtract_1": -status_weights[1].expand_as(pre_best_sample),
    "has_1_minus_0": sample_has[:, None] * (status_weights[1]-status_weights[0]),
}
status_rows = []
for status_label, delta_wc in status_deltas.items():
    ps = pre_best_sample + delta_wc @ W0.T
    for activation in ("silu", "gelu", "relu"):
        pp = torch.softmax(
            linear_repair(act_repair(ps, activation), "attr_heads.value_bin.2")
            + transition_layout[si], dim=-1
        )
        mm, nn, av = sample_metrics_repair(pp)
        status_rows.append((status_label, activation, mm, nn, av))
status_table_repair = pd.DataFrame(
    status_rows, columns=["status_delta", "activation", "mean_RMSE", "nll_RMSE", "mean_nll"]
).sort_values("nll_RMSE")
print("\n=== STATUS TESTS ===")
print(status_table_repair.head(10).round(6).to_string(index=False))

zhead_prefixes = [
    n[:-len(".weight")] for n in NAMES
    if n.endswith(".weight") and ("z_head" in n or "z_to_head" in n)
]
if len(zhead_prefixes) == 1:
    zhp = zhead_prefixes[0]
    view_mean_probe = context_layout[si] + linear_repair(prior_layout[si, :32], zhp)
    q_mean_probe = norm_repair(
        view_mean_probe + load("type_emb.weight")[1], "query_norm"
    )
    print("\nq reconstruction RMS:",
          float((q_mean_probe-q_layout[si]).square().mean().sqrt()))
    if float((q_mean_probe-q_layout[si]).square().mean().sqrt()) < 1e-4:
        gen_mc = torch.Generator().manual_seed(87221)
        draw_probs = []
        p_mu = prior_layout[si, :32]
        p_sd = prior_layout[si, 32:].clamp(-10, 5).exp()
        print("prior SD quantiles:",
              np.quantile(p_sd.numpy(), [0, .1, .5, .9, 1]).round(4))
        for draw in range(32):
            zs = p_mu + p_sd * torch.randn(p_mu.shape, generator=gen_mc)
            qs = norm_repair(
                context_layout[si] + linear_repair(zs, zhp)
                + load("type_emb.weight")[1], "query_norm"
            )
            ps = pre_best_sample + (qs-q_layout[si]) @ W0.T
            draw_probs.append(torch.softmax(
                linear_repair(act_repair(ps, ha), "attr_heads.value_bin.2")
                + transition_layout[si], dim=-1
            ))
        draws_mc_repair = torch.stack(draw_probs)
        p_mc32 = draws_mc_repair.mean(0)
        groups4 = draws_mc_repair.reshape(8, 4, len(si), 17).mean(1)
        means4 = (groups4 * sample_centres[None]).sum(-1)
        nlls4 = -torch.log(groups4.gather(
            2, sample_obsbin[None, :, None].expand(8, -1, -1)
        ).squeeze(-1).clamp_min(1e-12))
        print("\n=== PRIOR MONTE CARLO CHECK ===")
        print("32-draw metrics:", sample_metrics_repair(p_mc32))
        print("RMS four-draw mean variability:",
              float(means4.std(0).square().mean().sqrt()))
        print("RMS four-draw NLL variability:",
              float(nlls4.std(0).square().mean().sqrt()))
        print("four-draw reference RMSEs:",
              [tuple(round(x, 5) for x in sample_metrics_repair(p))
               for p in groups4])
else:
    print("No unique z-head prefix; Monte Carlo check skipped:", zhead_prefixes)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 16\n"
        "Verified that register ages are already log(seconds+1)/10. Tested "
        "status corrections with the best register layout and compared "
        "remaining head mismatch with four-draw prior variability when the "
        "latent-head prefix could be identified. No weights edited.\n"
    )
```