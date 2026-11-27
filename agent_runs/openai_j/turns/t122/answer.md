The verification stopped because the cached MLP helper requires an explicit activation argument; it did not test or contradict the new layout. I’ll rerun it with GELU, as specified in the architecture. Close agreement in sampled NLL, as well as means, would justify using these features for the repair.

```python
print("Helper signatures:", inspect.signature(mlp), inspect.signature(ln))
prior_parameters = mlp(verified_head["context"], "z_prior", "gelu")
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

print("\n=== MATCHED METADATA TABLES ===")
meta_groups = {}
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and {"state_i", "code", "zlast"}.issubset(obj.columns):
        meta_groups.setdefault(id(obj), {"names": [], "object": obj})["names"].append(name)
for group in meta_groups.values():
    obj = group["object"]
    print(group["names"], obj.shape, list(obj.columns))

with open("/work/notes.md", "a") as file:
    file.write("\n## Runs 107–109: register breakthrough\n"
               "- Measurement register: has_last at 0, log-age/10 at 3, last value at 6; other fields zero.\n"
               "- Deterministic scaled mean RMSE 0.0504, bias 0.00081; agreement on both stay halves is consistent with GPU prior-sampling noise.\n"
               "- Prior head-edit safety estimates used the wrong register and are invalid.\n"
               "- No active edits; 12 forward passes. Checking sampled likelihood and rebuilding contrast features.\n")
```
