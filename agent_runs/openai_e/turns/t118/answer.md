The posterior route did not improve the leakage–sensitivity tradeoff, and the local reconstruction is approximate. I won’t edit posterior weights. Instead, I’ll install a modest, reversible NE correction using the corrected recurrent features, balanced between doubling and halving. GPU factual fit will determine whether it is safe; rollouts will determine whether it moves both NE responses toward the record-derived targets.

```python
# Fit a balanced corrected recurrent detector; preserve baseline77 exactly.
DF103 = D95
train103 = fit_cf_mask93
test103 = test_cf_mask93.reshape(-1, 4).all(axis=1)
GF103 = XF95[fit_fact_mask93].T @ XF95[fit_fact_mask93] / fit_fact_mask93.sum()
G103, R103 = {}, {}
for key103, dx103 in DF103.items():
    y103 = np.repeat((key103[1] - 1.) * dose_ne96, 4)
    if key103[0] != "norepinephrine":
        y103 = np.zeros_like(y103)
    G103[key103] = dx103[train103].T @ dx103[train103] / train103.sum()
    R103[key103] = dx103[train103].T @ y103[train103] / train103.sum()

candidates103 = []
for half103 in (2.2, 2.6, 3.0, 3.4):
    A103 = (G103[("norepinephrine", 2.)]
            + half103 * G103[("norepinephrine", .5)]
            + 30. * (G103[("insulin", 2.)] + G103[("insulin", .5)])
            + 3000. * GF103)
    ridge103 = 1e-7 * np.trace(A103) / len(A103)
    theta103 = sla.solve(A103 + ridge103 * np.eye(len(A103)),
                        R103[("norepinephrine", 2.)]
                        + half103 * R103[("norepinephrine", .5)],
                        assume_a="pos")
    feature103 = (null95 @ theta103) / scale95
    slopes103 = {}
    for factor103 in (2., .5):
        pred103 = (DF103[("norepinephrine", factor103)] @ theta103).reshape(-1, 4).mean(axis=1)
        target103 = (factor103 - 1.) * dose_ne96
        select103 = test103 & (dose_ne96 > 0)
        slopes103[factor103] = np.dot(pred103[select103], target103[select103]) / np.dot(target103[select103], target103[select103])
    fact103 = XF95[test_fact_mask93] @ theta103
    ratio103 = slopes103[.5] / slopes103[2.]
    metric103 = {"half_w": half103, "s2": float(slopes103[2.]),
                 "s_half": float(slopes103[.5]), "half_ratio": float(ratio103),
                 "fact_per_x2": float(np.sqrt(np.mean(fact103**2)) / slopes103[2.])}
    candidates103.append((abs(ratio103 - 1.), feature103, metric103))

_, ne_feature103, ne_metrics103 = min(candidates103, key=lambda x103: x103[0])
ne_direction103 = np.load("/work/ne_candidate82.npz")["direction"]
beta_correction103 = 1.0  # Deliberately below the record-derived one-hour NE effect (~3).
strength103 = beta_correction103 / ne_metrics103["s2"]
baseline_delta103 = np.load("/work/baseline77.npz")["delta"]
ne_delta103 = strength103 * np.outer(ne_direction103, ne_feature103)
original_b103 = asnp101(load_original("core.b_in.weight"))
save_edit("core.b_in.weight",
          torch.as_tensor(original_b103 + baseline_delta103 + ne_delta103, dtype=torch.float32))
np.savez("/work/ne_trial103.npz", delta=ne_delta103, feature=ne_feature103,
         direction=ne_direction103, strength=strength103, beta=beta_correction103)

validation_stays103 = pd.read_csv("/work/fwd_9/factual_original.csv").stay_id.drop_duplicates().astype(int).tolist()
print("BALANCED CANDIDATES", [c103[2] for c103 in candidates103])
print("INSTALLED REVERSIBLE TRIAL", ne_metrics103,
      "strength", float(strength103), "NE delta norm", float(np.linalg.norm(ne_delta103)))
print("CURRENT EDITS", edited())
print("VALIDATION STAYS", validation_stays103)

with open("/work/notes.md", "a") as f103:
    f103.write("\n## Run 103: modest corrected balanced NE trial\n"
               "Posterior route rejected as no more selective. Local original reconstruction "
               "out RMSE=.0507, so GPU tests remain decisive. Installed baseline77 plus a "
               f"corrected balanced b_in NE rank-one trial, beta correction={beta_correction103}, "
               f"metrics={ne_metrics103}, NE delta norm={np.linalg.norm(ne_delta103):.3f}. "
               "Trial unaccepted pending factual and paired rollout tests. Saved ne_trial103.npz. "
               "Python runs: 103; forwards: 12.\n")
```