The deterministic candidate barely changes initial glucose, unlike the GPU result. The sampled calculation also omitted the code embedding, so it is not yet a valid comparison. I’ll restore that term; matching both GPU outputs would confirm that latent-noise sensitivity caused the failure.

```python
audit_code_embedding = wt("embed.code.weight")[
    audit_initial_base["code"]
]
print("query error after accounting for code embedding:",
      float((audit_q - (
          audit_initial_raw["inputs"]
          - audit_initial_raw["register"]
          - audit_code_embedding
      )).abs().max()))

audit_inputs_mc = (
    audit_q_mc
    + audit_initial_raw["register"][None]
    + audit_code_embedding[None]
)
audit_mc_original_probs = audit_head_logits(
    audit_inputs_mc, audit_original_weights, audit_transition[None]
).softmax(-1)
audit_mc_candidate_probs = audit_head_logits(
    audit_inputs_mc, r131_candidate, audit_transition[None]
).softmax(-1)
audit_mc_original = (
    audit_mc_original_probs * audit_centres[None]
).sum(-1)
audit_mc_candidate = (
    audit_mc_candidate_probs * audit_centres[None]
).sum(-1)
print(pd.DataFrame({
    "event": audit_initial_meta.event,
    "original_deterministic": audit_det_original.numpy(),
    "candidate_deterministic": audit_det_candidate.numpy(),
    "original_MC_mean": audit_mc_original.mean(0).numpy(),
    "candidate_MC_mean": audit_mc_candidate.mean(0).numpy(),
    "original_MC_sd": audit_mc_original.std(0).numpy(),
    "candidate_MC_sd": audit_mc_candidate.std(0).numpy()
}).to_string(index=False))

print("\n=== DYNAMIC ESTIMATES ===")
for filename in ("dynamic_iv.csv", "kernel_targets.csv", "sensor_iv.csv"):
    frame = pd.read_csv("/work/" + filename)
    print(filename, frame.shape)
    print(frame.head(8).to_string(index=False)[:1400])

print("\n=== IDENTIFICATION NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[5000:8500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 139: stochastic-head fingerprint\n"
            "- Corrected the MC audit to include the code embedding.\n"
            "- Deterministic candidate predictions are not a sufficient "
            "factual-safety test; comparing original and rejected outputs "
            "under the prior distribution.\n"
            "- No active edits; forward passes used: 14.\n")
```
