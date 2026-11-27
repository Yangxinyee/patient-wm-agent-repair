The posterior features reproduce the factor-one inputs exactly, so their alignment checks out. I’ll now fit dose detectors on them and verify the local state update against saved GPU states. Balanced held-out sensitivity, low factual leakage, and an accurate reconstruction would justify calibrating a posterior-weight edit; failure on any check would rule it out.

```python
# Posterior hidden-feature detector: closed-form regressions only.
train_bundle102 = fit_fact_mask93.reshape(-1, 4).all(axis=1)
test_bundle102 = test_fact_mask93.reshape(-1, 4).all(axis=1)
train_cf102 = fit_cf_mask93.reshape(-1, 4).all(axis=1)
test_cf102 = test_cf_mask93.reshape(-1, 4).all(axis=1)
scale_post102 = np.maximum(post_hidden101[train_bundle102].std(axis=0), 0.05)
XP102 = np.c_[post_hidden101 / scale_post102, np.ones(len(post_hidden101))]

stay_key102 = next(k102 for k102 in S0.keys() if "stay" in k102.lower())
stay_all102 = np.asarray(S0[stay_key102])
first102 = np.sort(np.unique(stay_all102, return_index=True)[1])
null_post102 = sla.null_space(XP102[first102], rcond=1e-7)
XPF102 = XP102 @ null_post102
GPF102 = XPF102[train_bundle102].T @ XPF102[train_bundle102] / train_bundle102.sum()

DP102, GP102, RP102 = {}, {}, {}
for key102, hidden102 in post_edit101.items():
    delta102 = np.c_[(hidden102 - post_hidden101[idx93]) / scale_post102,
                     np.zeros(len(idx93))]
    DP102[key102] = delta102 @ null_post102
    target102 = (key102[1] - 1.) * dose_ne96
    if key102[0] != "norepinephrine":
        target102 = np.zeros_like(target102)
    dx102 = DP102[key102][train_cf102]
    GP102[key102] = dx102.T @ dx102 / train_cf102.sum()
    RP102[key102] = dx102.T @ target102[train_cf102] / train_cf102.sum()

post_fits102 = {}
post_metrics102 = []
for half102 in (1., 4., 16.):
    for anchor102 in (30., 300., 3000.):
        cross102 = 100.
        A102 = (GP102[("norepinephrine", 2.)]
                + half102 * GP102[("norepinephrine", .5)]
                + cross102 * (GP102[("insulin", 2.)] + GP102[("insulin", .5)])
                + anchor102 * GPF102)
        ridge102 = 1e-7 * np.trace(A102) / len(A102)
        theta102 = sla.solve(A102 + ridge102 * np.eye(len(A102)),
                            RP102[("norepinephrine", 2.)]
                            + half102 * RP102[("norepinephrine", .5)],
                            assume_a="pos")
        raw102 = (null_post102 @ theta102) / np.r_[scale_post102, 1.]
        factual102 = XPF102 @ theta102
        m102 = {"half_w": half102, "anchor": anchor102,
                "fact_rms": float(np.sqrt(np.mean(factual102[test_bundle102]**2)))}
        for key102, dx102 in DP102.items():
            pred102 = dx102 @ theta102
            select102 = test_cf102 & (dose_ne96 > 0)
            if key102[0] == "norepinephrine":
                yy102 = (key102[1] - 1.) * dose_ne96
                m102[f"s{key102[1]}"] = float(
                    np.dot(yy102[select102], pred102[select102]) /
                    np.dot(yy102[select102], yy102[select102]))
            else:
                m102[f"ins_rms{key102[1]}"] = float(
                    np.sqrt(np.mean(pred102[test_cf102]**2)))
        m102["fact_per_x2"] = m102["fact_rms"] / max(abs(m102["s2.0"]), 1e-12)
        m102["half_ratio"] = m102["s0.5"] / max(m102["s2.0"], 1e-12)
        ne_test102 = test_cf102 & (dose_ne96 > 0)
        m102["NE_fact_per_x2"] = float(
            np.sqrt(np.mean(factual102[idx93[ne_test102]]**2)) /
            max(abs(m102["s2.0"]), 1e-12))
        post_fits102[(half102, anchor102)] = {"raw": raw102, "theta": theta102, "metrics": m102}
        post_metrics102.append(m102)

print("POSTERIOR NULLSPACE", null_post102.shape, "initial bundles", len(first102))
print(pd.DataFrame(post_metrics102).round(5).to_string(index=False))

# Identify and verify the retained local reconstruction without changing edits.
matches102 = []
for name102 in ("core_rows", "core_selected", "actual_rows45"):
    ix102 = np.asarray(globals()[name102], dtype=int)
    if len(ix102) == 32 and ix102.min() >= 0 and ix102.max() < len(hflow46):
        err102 = float(np.max(np.abs(asnp101(hflow46[ix102]) - asnp101(core_h_flow))))
        matches102.append((err102, name102))
print("\nLOCAL ROW MATCHES", matches102)
local_rows102 = np.asarray(globals()[min(matches102)[1]], dtype=int)
print("e/u/z errors relative to earlier local inputs",
      float(torch.max(torch.abs(e_fact94[local_rows102] - core_E_obs))),
      float(torch.max(torch.abs(u_fact94[local_rows102] - core_U_actions))),
      float(torch.max(torch.abs(z_true93[local_rows102] - core_z))))
h_orig102, o_orig102 = core_update_local(
    z_true93[local_rows102], core_config45,
    h_flow=core_h_flow, e=e_fact94[local_rows102], u=u_fact94[local_rows102],
    memory_data=core_memories[core_config45[3]]
)
print("LOCAL ORIGINAL RECONSTRUCTION RMSE h/out",
      float(torch.sqrt(torch.mean((h_orig102 - core_target_h)**2))),
      float(torch.sqrt(torch.mean((o_orig102 - core_target_out)**2))))

with open("/work/notes.md", "a") as f102:
    f102.write("\n## Run 102: posterior detector frontier and local reconstruction\n"
               "Factor-one posterior inputs are exact. Fitted initial-admission-null posterior "
               "detectors with factual and insulin guards, evaluated by held-out stays. "
               "Checked local recurrent reconstruction against original GPU states. "
               "No weights changed. Python runs: 102; forwards: 12.\n")
```