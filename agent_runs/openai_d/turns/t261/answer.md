The calibration gains are too small to rescue the combined edit. I’ll leave those transition weights unchanged and instead refit the norepinephrine direction to suppress factual-state noise while retaining its diagnostic probe sensitivity. This is a statistical fit, not model training. A candidate must lower independent factual loss, then retain its response to real dose edits.

```python
def r225_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

r225_sd = r225_np(r132_feat["sd"]).astype(float)
r225_P = r225_np(r132_feat["projection"]).astype(float)
r225_d = r225_np(r132_dmean).astype(float)

print("=== TRANSFORM AUDIT ===")
print("projection symmetry error:", float(np.max(np.abs(r225_P-r225_P.T))))
print("projection idempotence error:",
      float(np.max(np.abs(r225_P @ r225_P-r225_P))))
print("feature SD range:", float(r225_sd.min()), float(r225_sd.max()))

r225_response_rows = []
for ridge225, feature225 in r138_features.items():
    vv225 = r225_np(feature225["v"]).reshape(-1)
    r225_response_rows.append({
        "ridge": ridge225, "raw_vector_norm": float(np.linalg.norm(vv225)),
        "dot_d_times_sd": float(vv225 @ (r225_d*r225_sd)),
        "dot_d_over_sd": float(vv225 @ (r225_d/r225_sd)),
        "dot_d": float(vv225 @ r225_d)
    })
print(pd.DataFrame(r225_response_rows).round(6).to_string(index=False))

r225_reference = r225_np(r138_features[1e-5]["v"]).reshape(-1)
print("current difference-vector agreement with 0.2 * reference:",
      float(np.max(np.abs(r223_ne_v.numpy()-0.2*r225_reference))))

r225_candidates = {}
r225_results = []
r225_valid = (
    np.max(np.abs(r225_P-r225_P.T)) < 1e-5
    and np.max(np.abs(r225_P @ r225_P-r225_P)) < 1e-5
    and max(abs(row["dot_d_times_sd"]-1) for row in r225_response_rows) < 0.01
)
print("transform accepted:", r225_valid)

if r225_valid:
    fit225 = (
        r224_fit_mask
        & (c210_train["meta"]["event"].to_numpy() == "map_arterial")
        & r222_meta["train"]["norepinephrine_before"].to_numpy()
    )
    raw225 = c210_train["X"][:, fit225, :].numpy().reshape(-1, 192).astype(float)
    r225_mean = raw225.mean(0)
    z225 = ((raw225-r225_mean)/r225_sd) @ r225_P
    r225_cov = z225.T @ z225 / len(z225)

    for ridge225 in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        direction225 = np.linalg.solve(r225_cov + ridge225*np.eye(192), r225_d)
        direction225 /= r225_d @ direction225
        v225 = (r225_P @ direction225) / r225_sd
        b225 = -float(r225_mean @ v225)
        candidate225 = {
            k: torch.as_tensor(v, dtype=torch.float32).clone()
            for k, v in c206_candidates["combined_strong"].items()
        }
        candidate225["attr_heads.value_bin.0.weight"][48] = (
            r223_ne_common_v + torch.tensor(0.2*v225, dtype=torch.float32)
        )
        candidate225["attr_heads.value_bin.0.weight"][171] = (
            r223_ne_common_v - torch.tensor(0.2*v225, dtype=torch.float32)
        )
        candidate225["attr_heads.value_bin.0.bias"][48] = r223_ne_common_b + 0.2*b225
        candidate225["attr_heads.value_bin.0.bias"][171] = r223_ne_common_b - 0.2*b225
        r225_candidates[ridge225] = candidate225

        ll225, _ = c210_eval(candidate225, c210_train)
        lt225, _ = c210_eval(candidate225, c210_tune)
        dt225 = ll225 - np.asarray(c210_train["loss0"])
        du225 = lt225 - np.asarray(c210_tune["loss0"])
        r225_results.append({
            "ridge": ridge225, "vector_norm": float(np.linalg.norm(v225)),
            "probe_response": float(v225 @ (r225_d*r225_sd)),
            "train_fit": float(dt225[r224_fit_mask].mean()),
            "train_validation": float(dt225[r224_val_mask].mean()),
            "tune100": float(du225.mean())
        })

    print("\n=== FRESH COVARIANCE SCREEN ===")
    print(pd.DataFrame(r225_results).round(6).to_string(index=False))
else:
    print("No direction fitted: the old feature transform needs further recovery.")

print("No weights installed.")
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 225: fresh NE covariance-direction audit\n")
    f.write("Transition calibration does not rescue factual fit; none installed. "
            "Audited the old probe-coordinate transform and, if valid, fitted "
            "label-free minimum-variance NE directions on fresh training inputs.\n")
    f.write("Transform accepted: " + str(r225_valid) + "\n")
    if r225_results:
        f.write(pd.DataFrame(r225_results).to_string(index=False) + "\n")
```
