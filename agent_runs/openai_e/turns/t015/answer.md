The local decoder fits best with GELU, but still differs from the forward predictions by about 1–2 units on average. Prior sampling may explain part of that difference, so I won’t treat it as exact. Next I’ll test whether its features can recover dose deviations from the usual dosing policy on held-out stays. Strong recovery would support a residual-dose correction; weak recovery would argue for editing an earlier layer.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def dose_covariates(frame):
    return np.column_stack([
        frame["t" if n == "hour" else n].to_numpy(dtype=np.float32)
        for n in feature_names
    ])

train_policy_rows = panel[(panel.t >= 0) & (panel.t < 24)]
policy_rng = np.random.default_rng(1201)
train_policy_idx = policy_rng.choice(len(train_policy_rows), 260000, replace=False)
policy_sample = train_policy_rows.iloc[train_policy_idx]
policy_X = dose_covariates(policy_sample)
dose_policy = {}
for drug_name in ["insulin", "norepinephrine"]:
    est = HistGradientBoostingRegressor(
        max_iter=180, max_leaf_nodes=31, learning_rate=.07,
        min_samples_leaf=80, l2_regularization=2.,
        early_stopping=True, random_state=1201)
    est.fit(policy_X, policy_sample[drug_name].to_numpy())
    dose_policy[drug_name] = est
    print(drug_name, "policy iterations", est.n_iter_,
          "training residual SD",
          np.std(policy_sample[drug_name].to_numpy() - est.predict(policy_X)))

tpanel_probe = tpanel.copy()
tX = dose_covariates(tpanel_probe)
for drug_name, est in dose_policy.items():
    tpanel_probe[drug_name + "_expected"] = np.maximum(0., est.predict(tX))
    tpanel_probe[drug_name + "_residual"] = (
        tpanel_probe[drug_name] - tpanel_probe[drug_name + "_expected"])
tpanel_lookup = tpanel_probe.set_index(["stay_id", "t"])

probe_targets = np.zeros((len(head_meta), 2), dtype=np.float32)
probe_rawdose = np.zeros((len(head_meta), 2), dtype=np.float32)
for i, row in enumerate(head_meta.itertuples()):
    interval = int(round(row.hour)) - 1
    if interval < 0 or (row.stay_id, interval) not in tpanel_lookup.index:
        continue
    pr = tpanel_lookup.loc[(row.stay_id, interval)]
    domain = 0 if row.event.startswith("glucose") else 1
    drug_name = ["insulin", "norepinephrine"][domain]
    probe_targets[i, domain] = pr[drug_name + "_residual"]
    probe_rawdose[i, domain] = pr[drug_name]

probe_H = head_hidden.numpy()
probe_scale = np.maximum(probe_H.std(0), .03)
probe_Z = probe_H / probe_scale
probe_fold = (head_meta.stay_id.to_numpy() % 2).astype(int)
probe_domain = np.array([0 if e.startswith("glucose") else 1 for e in head_meta.event])
probe_results = []
for alpha in [.1, 1., 10., 100., 1000.]:
    pred_probe = np.zeros_like(probe_targets)
    pred_raw = np.zeros_like(probe_rawdose)
    for fold0 in [0, 1]:
        tr0, te0 = probe_fold != fold0, probe_fold == fold0
        rr = Ridge(alpha=alpha).fit(probe_Z[tr0], probe_targets[tr0])
        pred_probe[te0] = rr.predict(probe_Z[te0])
        rr2 = Ridge(alpha=alpha).fit(probe_Z[tr0], probe_rawdose[tr0])
        pred_raw[te0] = rr2.predict(probe_Z[te0])
    scores = []
    for j, drug_name in enumerate(["insulin", "norepinephrine"]):
        mask0 = (probe_domain == j) & (head_meta.hour.to_numpy() >= 1)
        other0 = probe_domain != j
        scores.append({
            "drug": drug_name,
            "residual_SD": round(float(probe_targets[mask0, j].std()), 4),
            "residual_R2": round(r2_score(probe_targets[mask0, j], pred_probe[mask0, j]), 4),
            "raw_dose_R2": round(r2_score(probe_rawdose[mask0, j], pred_raw[mask0, j]), 4),
            "other_event_leakage_RMS": round(float(np.sqrt(np.mean(pred_probe[other0, j]**2))), 4)
        })
    probe_results.append((alpha, scores))
print("\n=== HELD-OUT-STAY DOSE PROBES ===")
for row in probe_results:
    print(row)

# Quantify sensitivity to the latent sampling omitted in the mean decoder.
latent_rng = torch.Generator().manual_seed(1201)
ctx_base = q_head  # retained mean-query features; build sampled queries from the original context
gap0 = mlp0(torch.log1p(dt0) / 10, "gap", best_act)
ctx_head = state_x[:, :192] + norm0(flow_h.mean(1), "core.out_norm") + gap0
latent_hidden = []
latent_mean = []
for sample0 in range(8):
    zdraw = z_prior_head[:, :32] + torch.exp(z_prior_head[:, 32:]) * torch.randn(
        z_prior_head[:, :32].shape, generator=latent_rng)
    qdraw = norm0(ctx_head + linear0(zdraw, "z_head")
                 + W0["type_emb.weight"][1], "query_norm")
    md, hd, _ = head_calc(
        qdraw + head_static + F.linear(register8, W0["last_numeric.weight"]),
        kind=best_act)
    latent_hidden.append(hd.numpy())
    latent_mean.append(md.numpy())
latent_hidden = np.stack(latent_hidden)
latent_mean = np.stack(latent_mean)
print("\nLatent-draw reading-mean SD by event:")
for event0 in head_meta.event.unique():
    mm = head_meta.event.to_numpy() == event0
    print(event0, float(np.sqrt(np.mean(latent_mean[:, mm].var(0)))))
print("Eight-draw decoder RMSE against GPU:",
      float(np.sqrt(np.mean((latent_mean.mean(0) - head_meta.gpu_mean.to_numpy())**2))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 12: policy residual probes\n"
            "Fitted training-record dose-policy nuisance models (260000 sampled intervals). "
            "Cross-stay ridge probes test residual dose and raw-dose recovery from reading-head "
            "features; separately quantified prior-sampling noise. "
            f"Probe results: {probe_results}\n"
            "Python runs: 12. Forward requests: 3.\n")
```