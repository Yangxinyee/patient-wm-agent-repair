Much of the reconstruction error is consistent with latent sampling, but the head features decode dose innovations poorly on held-out stays. I’ll test a reversible alternative: remove action input from the latent posterior while retaining the direct action pathway. Stronger, correctly signed rollout effects with little factual-loss change would support the hypothesis that latent inference is masking drug responses.

```python
# Controlled, reversible ablation: only the posterior's action-input block.
# Context, observation input, action encoder, direct action pathway, and heads stay original.
name = "z_post.0.weight"
posterior_action_original = as_np(load_original(name)).copy()
posterior_no_action = posterior_action_original.copy()
posterior_no_action[:, 384:576] = 0.0
save_edit(name, posterior_no_action.astype(np.float32))

print("EDIT", name, "action-block norm before/after",
      np.linalg.norm(posterior_action_original[:, 384:576]),
      np.linalg.norm(posterior_no_action[:, 384:576]))
print("CURRENT EDITS", edited())
print("TUNING A IDS", target_ids_A.tolist())
print("CORE ATTENTION TENSORS")
for n in NAMES:
    if n.startswith(("core.prelude", "core.blocks.0.0", "core.coda",
                     "embed.categorical", "embed.kind", "embed.ekind", "encode.")):
        print(n, SHAPES[n])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 25: posterior-action ablation candidate\n"
            "- Head-only innovation decoding did not generalize (held-out R2 approximately zero).\n"
            "- Latent sampling accounts for much of the local-vs-GPU head discrepancy.\n"
            "- Exploratory edit: zero z_post.0.weight[:,384:576], retaining direct action input.\n"
            "- Hypothesis: action-conditioned latent severity inference masks direct drug responses.\n"
            "- Requires paired factual and rollout tests; this is not yet a validated repair.\n"
            "- Earlier small-sample error-vs-innovation SE printouts include an intercept SE.\n")
```
