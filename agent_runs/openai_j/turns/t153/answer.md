The audit stopped on a float32/float64 mismatch, not a model discrepancy. I’ll rerun with consistent types. The comparison will show whether latent sampling explains the rejected edit’s large initial glucose shift.

```python
def audit_head_logits(inputs, weights, transition):
    def cast(name):
        return torch.as_tensor(
            weights[name], dtype=inputs.dtype, device=inputs.device
        )
    hidden = F.gelu(F.linear(
        inputs, cast("attr_heads.value_bin.0.weight"),
        cast("attr_heads.value_bin.0.bias")
    ))
    return F.linear(
        hidden, cast("attr_heads.value_bin.2.weight"),
        cast("attr_heads.value_bin.2.bias")
    ) + transition

audit_transition = audit_initial_base["transition"]
audit_original_logits = audit_head_logits(
    audit_initial_base["inputs"], audit_original_weights, audit_transition
)
audit_candidate_logits = audit_head_logits(
    audit_initial_base["inputs"], r131_candidate, audit_transition
)
audit_det_original = (
    audit_original_logits.softmax(-1) * audit_centres
).sum(-1)
audit_det_candidate = (
    audit_candidate_logits.softmax(-1) * audit_centres
).sum(-1)

audit_context = audit_initial_base["context"]
audit_prior = F.linear(
    F.gelu(F.linear(audit_context, wt("z_prior.0.weight"),
                    wt("z_prior.0.bias"))),
    wt("z_prior.2.weight"), wt("z_prior.2.bias")
)
audit_zmean, audit_logsd = audit_prior.chunk(2, -1)
print("prior log-sd range:",
      float(audit_logsd.min()), float(audit_logsd.max()))

audit_q = F.layer_norm(
    audit_initial_base["view"] + wt("type_emb.weight")[0],
    (192,), wt("query_norm.weight"), wt("query_norm.bias")
)
audit_q_expected = (
    audit_initial_raw["inputs"] - audit_initial_raw["register"]
)
print("query reconstruction max error:",
      float((audit_q - audit_q_expected).abs().max()))

torch.manual_seed(138)
audit_draws = 512
audit_noise = torch.randn(audit_draws, 4, 32)
audit_view_mc = audit_initial_base["view"][None] + F.linear(
    audit_noise * audit_logsd.clamp(-6, 2).exp()[None],
    wt("z_head.weight")
)
audit_q_mc = F.layer_norm(
    audit_view_mc + wt("type_emb.weight")[0],
    (192,), wt("query_norm.weight"), wt("query_norm.bias")
)
audit_inputs_mc = audit_q_mc + audit_initial_raw["register"][None]
audit_mc_original = (
    audit_head_logits(audit_inputs_mc, audit_original_weights,
                      audit_transition[None]).softmax(-1)
    * audit_centres[None]
).sum(-1)
audit_mc_candidate = (
    audit_head_logits(audit_inputs_mc, r131_candidate,
                      audit_transition[None]).softmax(-1)
    * audit_centres[None]
).sum(-1)
print(pd.DataFrame({
    "event": audit_initial_meta.event,
    "original_deterministic": audit_det_original.numpy(),
    "candidate_deterministic": audit_det_candidate.numpy(),
    "original_MC_mean": audit_mc_original.mean(0).numpy(),
    "candidate_MC_mean": audit_mc_candidate.mean(0).numpy(),
    "candidate_MC_sd": audit_mc_candidate.std(0).numpy()
}).to_string(index=False))

print("GPU initial readings:")
print(tune_factual_original[
    (tune_factual_original.stay_id == 2020000000)
    & (tune_factual_original.hour == -1)
].merge(
    tune_factual_rejected,
    on=["stay_id", "hour", "event", "observed"],
    suffixes=("_original", "_candidate")
).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 133–138: matched-state audit\n"
            "- Original tuning states obtained in forward 14.\n"
            "- Train and tuning initial states match exactly.\n"
            "- Testing deterministic and sampled rejected-head predictions; "
            "dtype mismatch fixed. No active edits.\n")
```
