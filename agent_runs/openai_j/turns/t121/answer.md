One layout now matches the original GPU means at the observed sampling-noise level on both halves of the stays: history presence in column 0, age in column 3, and last value in column 6. Its NLL agreement is also substantially better. I’ll verify it using prior sampling and rebuild the action-contrast features with this layout. Residual systematic errors would refute the reconstruction.

```python
def corrected_register_head(meta, base):
    numbers = torch.zeros_like(base["numbers"])
    numbers[:, 0] = torch.tensor(meta.has_last.to_numpy(), dtype=torch.float32)
    numbers[:, 3] = torch.log1p(torch.tensor(meta.age_s.to_numpy(), dtype=torch.float32)) / 10
    numbers[:, 6] = torch.tensor(meta.zlast.to_numpy(), dtype=torch.float32).clamp(-10, 10)
    inputs = base["inputs"] + F.linear(
        numbers - base["numbers"], wt("last_numeric.weight")
    )
    hidden = F.gelu(F.linear(inputs, first_W, first_b))
    transition = wt("embed_transition.value_bin.weight")[
        torch.tensor(meta.code.to_numpy()*17 + meta.last_bin.to_numpy(), dtype=torch.long)
    ]
    logits = F.linear(hidden, last_W, last_b) + transition
    return {**base, "numbers": numbers, "inputs": inputs, "hidden": hidden,
            "logits": logits, "probs": logits.softmax(-1), "transition": transition}

verified_head = corrected_register_head(hm, audit_head)
prior_parameters = mlp(verified_head["context"], "z_prior")
print("Prior parameter shape:", tuple(prior_parameters.shape))
print("Prior log-SD range:", prior_parameters[:, 32:].min().item(),
      prior_parameters[:, 32:].max().item())

mc_generator = torch.Generator().manual_seed(108)
mc_noise = torch.randn((64, len(hm), 32), generator=mc_generator)
mc_delta_z = mc_noise * prior_parameters[:, 32:].exp()[None]
mc_views = verified_head["view"][None] + F.linear(mc_delta_z, wt("z_head.weight"))
mc_query = ln(mc_views + wt("type_emb.weight")[1], "query_norm")
det_query = ln(verified_head["view"] + wt("type_emb.weight")[1], "query_norm")
mc_inputs = mc_query + (verified_head["inputs"] - det_query)[None]
mc_hidden = F.gelu(F.linear(mc_inputs, first_W, first_b))
mc_logits = F.linear(mc_hidden, last_W, last_b) + verified_head["transition"][None]
mc_probs = mc_logits.softmax(-1)
mc_mean_probs = mc_probs.mean(0)
mc_means = (mc_mean_probs * audit_centres).sum(-1).numpy()
mc_groups = mc_probs.reshape(16, 4, len(hm), 17).mean(1)
mc_group_nll = -mc_groups[:, torch.arange(len(hm)), audit_targets].clamp_min(1e-12).log()
mc_expected_nll = mc_group_nll.mean(0).numpy()

mc_audit_table = pd.DataFrame({
    "event": hm.event,
    "scaled_mean_error": (mc_means-audit_original_mean)/audit_scale,
    "cpu_expected_nll": mc_expected_nll,
    "gpu_nll": audit_original_nll,
})
print("\n=== PRIOR-SAMPLED RECONSTRUCTION ===")
print(mc_audit_table.groupby("event").agg(
    mean_bias=("scaled_mean_error", "mean"),
    mean_rmse=("scaled_mean_error", lambda x: np.sqrt(np.mean(x**2))),
    cpu_nll=("cpu_expected_nll", "mean"),
    gpu_nll=("gpu_nll", "mean")
).to_string())
print("Overall CPU expected four-draw NLL:", mc_expected_nll.mean())

print("\n=== CACHED COUNTERFACTUAL HEADS ===")
for key, head in repair_readout_cf.items():
    print(key, {k: tuple(v.shape) for k, v in head.items()
                if torch.is_tensor(v) or isinstance(v, np.ndarray)})

print("\n=== MATCHED METADATA TABLES (grouped by object) ===")
meta_groups = {}
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and {"state_i", "code", "zlast"}.issubset(obj.columns):
        meta_groups.setdefault(id(obj), {"names": [], "object": obj})["names"].append(name)
for group in meta_groups.values():
    obj = group["object"]
    print(group["names"], obj.shape, list(obj.columns))

with open("/work/notes.md", "a") as file:
    file.write("\n## Run 107 breakthrough; run 108 validation\n"
               "- Eight-input measurement register is consistent with has_last at 0, age at 3, last value at 6; other inputs zero.\n"
               "- Deterministic mean scaled RMSE 0.0504 and bias 0.00081, matching measured GPU sampling noise; both stay halves agree.\n"
               "- Previous head-edit safety estimates used the wrong register and are invalid.\n"
               "- Validating four-prior-draw likelihood before refitting any repair. No active edits; 12 forward passes.\n")
```
