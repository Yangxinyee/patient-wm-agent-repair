Latent amplification barely changed the rollouts, so it is not a useful repair at this strength. I’ll remove it and inspect the saved `out_norm` designs, which act on the state summary read by the heads. A useful pilot would change dose responses substantially more than factual predictions.

```python
repair_candidates["latent_in_gain21"] = freeze_ops(ops())
replay_ops(repair_candidates["numeric_gain4"])

print("OUT-NORM PARAMETER ESTIMATES")
for label, vector in [("gamma", norm_gamma_est), ("beta", norm_beta_est),
                      ("out_mean", out_mu)]:
    print(label, "norm", float(np.linalg.norm(vector)),
          "quantiles", np.quantile(vector, [0, .1, .5, .9, 1]).tolist())

print("\nSAVED DIAGONAL DESIGNS")
for penalty, design_item in diag_designs.items():
    print("penalty", penalty)
    if isinstance(design_item, dict):
        for key, value in design_item.items():
            if isinstance(value, np.ndarray):
                print(" ", key, value.shape, "norm", float(np.linalg.norm(value)),
                      "range", [float(value.min()), float(value.max())],
                      "values", value.tolist() if value.size <= 10 else "")
            elif np.isscalar(value):
                print(" ", key, value)
            else:
                print(" ", key, type(value).__name__)
    else:
        print(brief_value(design_item))

print("\nHEAD SURROGATES")
for event, saved_fit in head_surrogates.items():
    print(event, "fields", list(saved_fit))
    for key in ["ridge", "cv_rmse", "penalty"]:
        if key in saved_fit:
            print(key, brief_value(saved_fit[key]))
    print("world coefficient norm", float(np.linalg.norm(saved_fit["w"][:192])))

print("\nDIAGONAL CONSTRAINTS")
print("Bdiag", Bdiag.shape)
print("target-related small arrays")
for key in ["target", "targets", "diag_target", "diag_targets", "desired", "rhs"]:
    if key in globals():
        print(key, brief_value(globals()[key]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 41 / forwards 20-21: latent candidate rejected\n"
            "- z_in latent gain 21: factual increase +0.00122, rollouts insulin -2.494/+1.741 and NE +0.123/+0.017.\n"
            "- No material gain over numeric gain 4. Saved candidate and restored numeric gain 4 alone.\n"
            "- Inspecting direct out_norm diagonal designs. Forward passes used: 21.\n")
```