This is still the soft-feature result: safe corrections gave only tiny dose-response gains. It does not yet show whether calibration can support a stronger MAP edit. I’ll run that combination screen, or display it if already cached. Lower factual loss at a higher MAP gain would justify a GPU test.

```python
if "r265_frontier" not in globals() or len(r265_frontier) != 48:
    r265_check = make242(0.90, 7.0)
    r265_match = max(float((r265_check[k]-weights259[k]).abs().max())
                     for k in weights259)
    print("Current-head reconstruction error:", r265_match)
    assert r265_match < 1e-5

    r265_calibrators = {
        "none": np.zeros(193),
        "soft_half": r264_betas["r0.1_p0.01_s0.5"],
        "soft_low": r264_betas["r0.1_p0.003_s1"],
        "soft_ridge": r264_betas["r1_p0.01_s1"],
    }
    r265_rows, r265_candidates = [], {}
    for r265_insulin in [0.86, 0.90, 0.94]:
        for r265_gain in [7.0, 8.5, 10.0, 12.0]:
            for r265_calname, r265_beta in r265_calibrators.items():
                r265_w = {k: v.clone() for k, v in
                          make242(r265_insulin, r265_gain).items()}
                r265_bt = torch.as_tensor(r265_beta, dtype=torch.float32)
                r265_w["attr_heads.value_bin.2.weight"] += (
                    v259[:, None]*(r265_bt[:192]/scale259)[None, :])
                r265_w["attr_heads.value_bin.2.bias"] += v259*(
                    r265_bt[-1] -
                    (r265_bt[:192]*center259/scale259).sum())
                r265_key = f"i{r265_insulin:g}_g{r265_gain:g}_{r265_calname}"
                r265_candidates[r265_key] = r265_w
                r265_row = {"key": r265_key, "insulin": r265_insulin,
                            "MAP_gain": r265_gain,
                            "calibrator": r265_calname}
                for r265_split in ["train", "tune"]:
                    r265_mc = packs259[r265_split]["mc"]
                    r265_pred = head177_eval(
                        r265_mc, r265_w, r265_mc["centres"])
                    r265_delta = (r265_pred["nll"] -
                                  np.asarray(r265_mc["nll"].mean(0)))
                    if r265_split == "train":
                        r265_row["fit_dNLL"] = float(r265_delta[fit259].mean())
                        r265_row["cv_dNLL"] = float(r265_delta[cv259].mean())
                        for r265_event in sensors259:
                            r265_mask = (np.asarray(cv259, bool) &
                                         (events259 == r265_event))
                            r265_row["cv_"+r265_event] = float(
                                r265_delta[r265_mask].mean())
                    else:
                        r265_row["tune_dNLL"] = float(r265_delta.mean())
                r265_rows.append(r265_row)
    r265_frontier = pd.DataFrame(r265_rows)
    r265_frontier.to_csv(
        "/work/calibration_strength_frontier265.csv", index=False)

r265_safe = r265_frontier[
    (r265_frontier.cv_dNLL <= 0.0030) &
    (r265_frontier.tune_dNLL <= 0.0040)
].sort_values(["MAP_gain", "tune_dNLL"], ascending=[False, True])
r265_cols = ["key", "fit_dNLL", "cv_dNLL", "tune_dNLL",
             "cv_glucose_finger", "cv_glucose_lab",
             "cv_map_arterial", "cv_map_cuff"]
print("Locally feasible combinations:", len(r265_safe))
print(r265_safe[r265_cols].head(16).to_string(index=False))
print("Current edits unchanged:", edited())

with open("/work/notes.md", "a") as r265_f:
    r265_f.write(
        "\n## Calibration/strength combination screen\n"
        "- Screening statistical hidden-feature calibration combined with "
        "independent insulin/MAP strengths; exact cached factual likelihood.\n"
        "- Results saved in calibration_strength_frontier265.csv. "
        "No weights changed; candidates require GPU validation.\n"
    )
```