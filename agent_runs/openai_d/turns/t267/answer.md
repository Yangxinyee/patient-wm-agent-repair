The disputed transform belongs to the guard regression, not the NE correction direction. The existing stronger edit predicts about −14.4/+6.6 for insulin and +4.7/−1.2 for NE, but exceeds the factual budget on one cohort. I’ll screen raw-coordinate NE directions with the same probe response and lower factual variance. Lower held-out loss would support trying them; actual dose rollouts must still verify their effects.

```python
r230_base = c207_tensor_weights(c206_candidates["combined_strong"])
r230_rows = (48, 171)
r230_common_v = ((r230_base[wn][48] + r230_base[wn][171]) / 2).numpy().astype(np.float64)
r230_common_b = float((r230_base[bn][48] + r230_base[bn][171]) / 2)
r230_old_v = ((r230_base[wn][48] - r230_base[wn][171]) / 2).numpy().astype(np.float64)
r230_old_b = float((r230_base[bn][48] - r230_base[bn][171]) / 2)
r230_d = np.asarray(r177_probe_direction, dtype=np.float64)
r230_target = float(r230_old_v @ r230_d)

print("=== RAW COORDINATE CHECK ===")
print("reference vs old probe fit:",
      np.max(np.abs(r225_reference - r138_features[1e-5]["v"])))
print("old signal norm / probe response:", np.linalg.norm(r230_old_v), r230_target)
print("common and signal biases:", r230_common_b, r230_old_b)

r230_events = c210_train["meta"]["event"].to_numpy()
r230_fit = np.asarray(r137_fit, dtype=bool)
r230_val = np.asarray(r137_val, dtype=bool)
r230_art_fit = r230_fit & (r230_events == "map_arterial")
r230_raw = c210_train["X"][:, r230_art_fit].numpy().reshape(-1,192).astype(np.float64)
r230_mean = r230_raw.mean(0)
r230_centered = r230_raw - r230_mean
r230_C = r230_centered.T @ r230_centered / len(r230_raw)
r230_variance_scale = np.trace(r230_C) / 192

r230_candidates = {}
r230_screen = []
for ridge in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    v = np.linalg.solve(r230_C + ridge*r230_variance_scale*np.eye(192), r230_d)
    v *= r230_target / (v @ r230_d)
    b = -float(r230_mean @ v)
    weights = {n: t.clone() for n,t in r230_base.items()}
    weights[wn][48] = torch.tensor(r230_common_v + v, dtype=torch.float32)
    weights[wn][171] = torch.tensor(r230_common_v - v, dtype=torch.float32)
    weights[bn][48] = r230_common_b + b
    weights[bn][171] = r230_common_b - b
    r230_candidates[ridge] = weights
    train_loss, _ = c210_eval(weights, c210_train)
    tune_loss, _ = c210_eval(weights, c210_tune)
    dl = train_loss-c210_train["loss0"]
    dt = tune_loss-c210_tune["loss0"]
    r230_screen.append(dict(ridge=ridge, norm=np.linalg.norm(v),
                           signal_sd=np.sqrt(v@r230_C@v),
                           fit=float(dl[r230_fit].mean()),
                           validation=float(dl[r230_val].mean()),
                           tune100=float(dt.mean())))
print(pd.DataFrame(r230_screen).to_string(index=False))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 230: raw NE covariance screen\n"
            "Recovered the guard mapping: r177_v = P@(beta/sd); it is orthogonal to the raw probe direction. "
            "This is separate from the actual NE correction direction. Screened training-only, "
            "closed-form minimum-variance raw directions preserving the installed NE probe response; "
            "no weights installed. Probe preservation is not proof of real dose-edit preservation.\n")
    f.write(pd.DataFrame(r230_screen).to_string(index=False)+"\n")
```
