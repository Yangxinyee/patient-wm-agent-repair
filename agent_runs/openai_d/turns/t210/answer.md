The cached inputs match, and the trial changes only the intended neuron pair and its output strength. The head helper returns MLP logits without the register’s transition logits; omitting those could explain the mismatch. I’ll test that directly. If adding the transition term removes the error, the trial’s local loss can be recomputed reliably.

```python
print("=== TEST FOR OMITTED TRANSITION LOGITS ===")
c182_transition = c180_np(r136_transition)
print("Transition shape:", c182_transition.shape,
      "maximum magnitude:", float(np.max(np.abs(c182_transition))))

for label, trial_logits in [
    ("unchanged", r178_logits),
    ("plus_transition", r178_logits + c182_transition),
    ("minus_transition", r178_logits - c182_transition),
]:
    print(label, "reconstruction maximum error:",
          float(np.max(np.abs(trial_logits - r178_expected))))

c182_checked_logits = c180_np(r145_head(r171_combined_backup)) + c182_transition
print("Checked candidate vs r174_logits:",
      float(np.max(np.abs(c182_checked_logits - r174_logits))))

c182_candidate_logits = c180_np(r145_head(r178_candidate)) + c182_transition
c182_error = float(np.max(np.abs(c182_candidate_logits - r178_expected)))
print("Fresh direct candidate reconstruction error:", c182_error)

print("=== LOSS BASELINES ===")
for name in ["r138_original_loss", "r138_loss0", "r175_base_loss", "r178_loss"]:
    if name in globals():
        arr = c180_np(globals()[name])
        print(name, arr.shape, "mean", float(arr.mean()))

c182_candidate_loss = r138_nll(c182_candidate_logits)
c182_checked_loss = r138_nll(c182_checked_logits)
c182_original_loss = c180_np(r138_original_loss)
c182_rows = []
for label, loss in [("checked_combined", c182_checked_loss),
                    ("corrected_trial", c182_candidate_loss)]:
    delta = loss - c182_original_loss
    row = {"candidate": label, "all": float(delta.mean()),
           "fit": float(delta[r177_fit].mean()),
           "validation": float(delta[r177_val].mean()),
           "by_event": {event: float(delta[r176_events == event].mean())
                        for event in np.unique(r176_events)}}
    c182_rows.append(row)
    print(row)

print("=== GATE SUMMARY ===")
for name in ["r177_new_G", "r177_new_pair", "r175_g", "r175_d", "r177_score"]:
    arr = c180_np(globals()[name])
    print(name, "shape", arr.shape,
          "quantiles", np.quantile(arr, [0, .1, .5, .9, 1]))
print("Trial output multiplier:", r178_strength)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 182: transition-logit audit\n"
            f"Fresh trial reconstruction after adding the register transition "
            f"term has maximum error {c182_error:.8g}. "
            f"Recomputed local diagnostics: {c182_rows}. "
            "No trial installed; checked combined fallback remains active.\n")
```