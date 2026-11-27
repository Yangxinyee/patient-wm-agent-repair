The exact-constraint corrections failed: they sharply worsened factual loss and did not improve held-out doubling sensitivity. None was installed. I’ll try one softer, regularized screen using dose-weighted moments. If it cannot produce a held-out gain at very small factual cost, I’ll abandon this approach.

```python
r264_M = []
for r264_label in labels259:
    r264_cf = packs259[r264_label[:2]]
    r264_x = np.asarray(r264_cf["mc"]["delta_rate"], float)
    r264_mask = r263_masks[(r264_label, "fit")]
    r264_diff = r264_cf["response_features"] - r263_base["response_features"]
    r264_M.append((r264_diff[r264_mask] * r264_x[r264_mask, None]).sum(0) /
                  np.square(r264_x[r264_mask]).sum())
r264_M = np.asarray(r264_M)
print("Dose-weighted design reconstruction error:",
      float(np.abs(r264_M-np.asarray(rows259)).max()))

r264_B = np.vstack([r264_M, r263_cal])
r264_bn = np.maximum(np.linalg.norm(r264_B, axis=1), 1e-8)
r264_Bn = r264_B/r264_bn[:, None]
r264_y = np.zeros(len(r264_B))
r264_y[[14, 15]] = 1.0
r264_y[[10, 11]] = 0.2
r264_yn = r264_y/r264_bn

r264_proposals = []
r264_betas = {}
for r264_ridge in [0.001, 0.1, 1.0]:
    for r264_penalty in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
        r264_b = np.linalg.solve(
            r263_H + r264_ridge*r263_hscale*np.eye(193) +
            r264_penalty*(r264_Bn.T@r264_Bn),
            r264_penalty*(r264_Bn.T@r264_yn))
        for r264_strength in [0.5, 1.0, 2.0]:
            r264_key = f"r{r264_ridge:g}_p{r264_penalty:g}_s{r264_strength:g}"
            r264_beta = r264_strength*r264_b
            r264_m = r264_M@r264_beta
            r264_cost = float(grad259@r264_beta +
                              0.5*r264_beta@r263_H@r264_beta)
            r264_betas[r264_key] = r264_beta
            r264_proposals.append({
                "key": r264_key, "pred_cost": r264_cost,
                "pred_A2": r264_m[14], "pred_C2": r264_m[15],
                "priority": (r264_m[14]+r264_m[15])/2 -
                            abs(r264_m[10]-0.2) -
                            abs(r264_m[11]-0.2) -
                            max(abs(r264_m[i]) for i in [0,1,4,5,8,9,12,13])
            })
r264_proposals = pd.DataFrame(r264_proposals)
r264_short = r264_proposals[
    (r264_proposals.pred_cost < 0.008) &
    (r264_proposals.pred_A2 > 0)
].sort_values("priority", ascending=False).head(28)

r264_rows = []
for r264_row in r264_short.to_dict("records"):
    r264_result = r263_evaluate(r264_betas[r264_row["key"]])
    r264_row.update({k: r264_result[k]
                    for k in ["fit_dNLL", "cv_dNLL", "tune_dNLL"]})
    for r264_factor in [0.5, 2.0]:
        for r264_event, r264_shortname in [
            ("map_arterial", "A"), ("map_cuff", "C")]:
            r264_row[f"cv_{r264_shortname}{r264_factor}"] = \
                r264_result[(("norepinephrine", r264_factor, r264_event), "cv")]
    r264_row["max_glucose_cv_change"] = max(
        abs(r264_result[(label, "cv")]) for label in labels259
        if label[2].startswith("glucose"))
    r264_rows.append(r264_row)

r264_frontier = pd.DataFrame(r264_rows)
r264_frontier.to_csv("/work/soft_hidden_frontier264.csv", index=False)
np.savez("/work/soft_hidden_betas264.npz", **r264_betas)
r264_safe = r264_frontier[
    (r264_frontier.cv_dNLL <= 0.0028) &
    (r264_frontier.tune_dNLL <= 0.0037) &
    (r264_frontier.max_glucose_cv_change <= 0.35)
].sort_values("cv_A2.0", ascending=False)
print("Screened:", len(r264_frontier), "locally safe:", len(r264_safe))
r264_cols = ["key", "fit_dNLL", "cv_dNLL", "tune_dNLL",
             "cv_A2.0", "cv_C2.0", "cv_A0.5", "cv_C0.5",
             "max_glucose_cv_change"]
print(r264_safe[r264_cols].head(12).to_string(index=False))
if len(r264_safe) == 0:
    print(r264_frontier.sort_values("tune_dNLL")[r264_cols]
          .head(5).to_string(index=False))
with open("/work/notes.md", "a") as r264_f:
    r264_f.write(
        "\n## Run 264: soft frozen-feature screen\n"
        "- Exact moment constraints failed badly on held-out factual loss; "
        "those candidates were not installed.\n"
        "- Screened regularized statistical moment regressions with soft "
        "constraints; results saved in soft_hidden_frontier264.csv.\n"
        "- No weights changed.\n"
    )
```