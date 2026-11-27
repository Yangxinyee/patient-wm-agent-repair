The lowest-penalty diagonal design has the smallest estimated factual cost: at 10% strength, it changes predicted means by about 0.62 glucose units and 0.18 MAP units in standard deviation. I’ll install that pilot, compensating its average state shift with a bias edit. Lower factual NLL degradation and stronger drug rollouts would support the design; failure of either check would reject it.

```python
replay_ops(repair_candidates["numeric_gain4"])
diag_pilot_strength = 0.10
diag_pilot_vector = np.array(diag_designs[1e-4], copy=True)
diag_gamma_new = norm_gamma_est * (1 + diag_pilot_strength*diag_pilot_vector)
diag_beta_new = norm_beta_est - diag_pilot_strength*diag_pilot_vector*(out_mu-norm_beta_est)

def change_inferred_vector(name, old_estimate, new_estimate):
    change_vec = np.asarray(new_estimate)-np.asarray(old_estimate)
    denom = float(change_vec @ old_estimate)
    if np.linalg.norm(change_vec) == 0:
        return
    if abs(denom) < 1e-12:
        raise ValueError("Ill-conditioned inferred-vector edit")
    alpha = float((change_vec @ change_vec)/denom)
    rank1(name, change_vec, alpha)
    print(name, "delta_norm", float(np.linalg.norm(change_vec)), "alpha", alpha)

change_inferred_vector("core.out_norm.weight", norm_gamma_est, diag_gamma_new)
change_inferred_vector("core.out_norm.bias", norm_beta_est, diag_beta_new)
repair_candidates["outnorm_diag_010"] = freeze_ops()
print("gamma multiplier range", float((1+.1*diag_pilot_vector).min()),
      float((1+.1*diag_pilot_vector).max()))
print("normalization reconstruction max error",
      float(np.max(np.abs(out_reconstructed-out_states))))
print("operation count", len(ops()))
with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 46: centered final-state diagonal pilot\n"
            "- Installed numeric gain 4 plus a 10%-strength diagonal out_norm design.\n"
            "- out_norm gamma/beta were inferred from observed h/out with near-exact reconstruction, not read from weights.\n"
            "- Bias compensates the average state shift; linear dose-response targets are -0.15 insulin/glucose and +0.25 NE/MAP per rate unit.\n"
            "- Pilot requires factual and rollout validation. Forward passes used: 21.\n")
```
