Replacing the two units can be compensated to numerical precision, and the gated decoder retains its controlled-dose response. The reported “inactive” RMS may include drug bundles without innovation labels, so I’ll check the actual action flags. I’ll also probe the 32 latent directions: consistent glucose sensitivity with little MAP sensitivity would support a selective insulin edit.

```python
actual_insulin95 = np.asarray(repair_bank["has_insulin"])[idx82].astype(bool)
print("True non-insulin gated RMS:",
      np.sqrt(np.mean(g94[test94 & ~actual_insulin95] ** 2)))
print("Insulin bundles, labelled bundles:",
      actual_insulin95.sum(), np.sum(weights82["insulin"] > 0))

rp_mlp_original95 = rp_mlp

def transition_with_z95(indices, delta):
    indices = np.asarray(indices, dtype=int)
    delta = torch.as_tensor(delta, dtype=torch.float32)
    def injected_mlp(x, prefix):
        result = rp_mlp_original95(x, prefix)
        if prefix == "z_post":
            result = result.clone()
            result[..., :32] += delta
        return result
    previous = globals()["rp_mlp"]
    globals()["rp_mlp"] = injected_mlp
    try:
        return transition73(indices)
    finally:
        globals()["rp_mlp"] = previous

drug_indices95 = idx82[
    np.asarray(repair_bank["has_insulin"])[idx82]
    | np.asarray(repair_bank["has_norepinephrine"])[idx82]
]
available95 = head_train_data_repair[
    head_train_data_repair["prev"].isin(drug_indices95)
]
rng95 = np.random.default_rng(9501)
choices95 = np.unique(available95["prev"].to_numpy())
probe_indices95 = np.sort(rng95.choice(
    choices95, size=min(256, len(choices95)), replace=False
))
probe_data95 = available95[
    available95["prev"].isin(probe_indices95)
].reset_index(drop=True)
centres95 = np.stack([
    centres_by_event74[event] for event in probe_data95["event"]
])

def probe_means95(state_values):
    bank = dict(repair_bank)
    bank["x"] = np.asarray(repair_bank["x"]).copy()
    bank["x"][probe_indices95] = np.asarray(state_values)
    _, probability = head_features_fixed(bank, probe_data95, draws=4, seed=9502)
    return np.sum(probability * centres95, axis=1)

base_transition95 = transition73(probe_indices95)
print("Probe bundles/readings:", len(probe_indices95), len(probe_data95))
print("Reconstruction RMSE:",
      np.sqrt(np.mean((np.asarray(base_transition95) -
                       repair_bank["x"][probe_indices95]) ** 2)))

J95 = np.zeros((len(probe_data95), 32), dtype=np.float64)
step95 = 0.10
for j in range(32):
    delta = np.zeros(32, dtype=np.float32)
    delta[j] = step95
    plus = probe_means95(transition_with_z95(probe_indices95, delta))
    minus = probe_means95(transition_with_z95(probe_indices95, -delta))
    J95[:, j] = (plus - minus) / (2 * step95)
    if (j + 1) % 8 == 0:
        print("Completed latent directions", j + 1)

is_glucose95 = probe_data95["event"].str.startswith("glucose").to_numpy()
scale95 = np.where(is_glucose95, 10.0, 3.0)
target95 = np.where(is_glucose95, -2.04924, 0.0)
fit_by_index95 = dict(zip(idx82.tolist(), train94.tolist()))
probe_fit95 = np.array([fit_by_index95[int(i)] for i in probe_data95["prev"]])
A95 = J95 / scale95[:, None]
b95 = target95 / scale95
directions95, rows95 = {}, []
for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
    v = np.linalg.solve(
        A95[probe_fit95].T @ A95[probe_fit95] + alpha * np.eye(32),
        A95[probe_fit95].T @ b95[probe_fit95]
    )
    directions95[alpha] = v
    response = J95 @ v
    row = {"alpha": alpha, "direction_norm": np.linalg.norm(v)}
    for label, mask in [
        ("glucose", ~probe_fit95 & is_glucose95),
        ("MAP", ~probe_fit95 & ~is_glucose95)
    ]:
        row[label + "_mean"] = np.mean(response[mask]) if mask.any() else np.nan
        row[label + "_sd"] = np.std(response[mask]) if mask.any() else np.nan
    rows95.append(row)
print(pd.DataFrame(rows95).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 95\n"
            "Two posterior hidden units can be repurposed with compensated mean-latent "
            "error about 2e-7 RMS. Checking action-flag leakage and finite-difference "
            "latent-to-reading sensitivities to derive an insulin-specific response "
            "direction by closed-form regression. No weights edited.\n")
```
