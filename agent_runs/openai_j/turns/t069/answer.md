The fit stopped because some reconstructed features contain NaNs, likely at stay boundaries where no preceding state exists. This is a reconstruction issue, not evidence against the dose signal. I’ll identify and exclude those rows—not impute them—and rerun the held-out test. Nonfinite features elsewhere would require further investigation.

```python
post_valid = (np.asarray(repair_valid, dtype=bool)
              & np.isfinite(post_H).all(axis=1)
              & np.isfinite(post_innov).all(axis=1))
for cf_h in post_cf.values():
    post_valid &= np.isfinite(cf_h).all(axis=1)
print("Valid bundles:", int(post_valid.sum()), "/", repair_n)
print("Excluded rows:",
      repair_sf.loc[~post_valid, ["stay_id", "hour"]].head(8).to_dict("records"))
print("Excluded active bundles:",
      int(((post_rates > 0).any(1) & ~post_valid).sum()))

def fit_post_contrast(drug, dj, mode, factual_weight, alpha, fitting_mask):
    fitting_mask = fitting_mask & post_valid
    active = (post_rates[:, dj] > 0) & fitting_mask
    target0 = np.zeros(repair_n) if mode == "zero" else post_innov[:, dj]
    xx = [post_H[fitting_mask]]
    yy = [target0[fitting_mask]]
    ww = [np.full(fitting_mask.sum(), factual_weight)]
    for factor in post_train_factors:
        xx.append(post_cf[(drug, factor)][active])
        yy.append(target0[active] + (factor - 1) * post_rates[active, dj])
        ww.append(np.ones(active.sum()))
    xx, yy, ww = np.vstack(xx), np.concatenate(yy), np.concatenate(ww)
    xm = xx.mean(0)
    xs = np.maximum(xx.std(0), 0.05)
    reg = Ridge(alpha=alpha).fit((xx - xm) / xs, yy, sample_weight=ww)
    coef = reg.coef_ / xs
    return coef, float(reg.intercept_ - coef @ xm)

post_cv_rows, post_full_models, post_predictions = [], {}, {}
for dj, drug in enumerate(["insulin", "norepinephrine"]):
    active = (post_rates[:, dj] > 0) & post_valid
    off = (post_rates[:, dj] == 0) & post_valid
    for mode in ["zero", "innovation"]:
        for factual_weight in [2.0, 10.0]:
            for alpha in [0.1, 10.0, 100.0]:
                key = (drug, mode, factual_weight, alpha)
                pf = np.full(repair_n, np.nan)
                pc = {f: np.full(repair_n, np.nan) for f in post_train_factors}
                for held_fold in [0, 1]:
                    fit_mask = (post_fold != held_fold) & post_valid
                    test_mask = (post_fold == held_fold) & post_valid
                    coef, intercept = fit_post_contrast(
                        drug, dj, mode, factual_weight, alpha, fit_mask)
                    pf[test_mask] = post_H[test_mask] @ coef + intercept
                    for factor in post_train_factors:
                        pc[factor][test_mask] = (
                            post_cf[(drug, factor)][test_mask] @ coef + intercept)
                row = dict(drug=drug, mode=mode, factual_weight=factual_weight,
                           alpha=alpha,
                           factual_rms=float(np.sqrt(np.mean(pf[post_valid]**2))),
                           active_rms=float(np.sqrt(np.mean(pf[active]**2))),
                           off_action_rms=float(np.sqrt(np.mean(pf[off]**2))))
                for factor, label in [(0.0, "zero"), (0.5, "half"),
                                      (2.0, "double"), (3.0, "triple")]:
                    true_delta = (factor - 1) * post_rates[active, dj]
                    pred_delta = pc[factor][active] - pf[active]
                    row["r2_" + label] = r2_score(true_delta, pred_delta)
                    row["gain_" + label] = float(
                        pred_delta @ true_delta / (true_delta @ true_delta))
                post_cv_rows.append(row)
                post_predictions[key] = (pf, pc)
                post_full_models[key] = fit_post_contrast(
                    drug, dj, mode, factual_weight, alpha, post_valid)

post_cv_table = pd.DataFrame(post_cv_rows)
print(post_cv_table.round(4).to_string(index=False))
post_cv_table.to_csv("/work/posterior_contrast_cv.csv", index=False)

post_latent_directions = {}
for dj, drug in enumerate(["insulin", "norepinephrine"]):
    active = (post_rates[:, dj] > 0) & post_valid
    dh = np.vstack([post_cf[(drug, f)][active] - post_H[active]
                    for f in post_train_factors])
    dr = np.concatenate([(f - 1) * post_rates[active, dj]
                         for f in post_train_factors])
    dz = dh @ wt("z_post.2.weight")[:32].numpy().T
    direction = (dr @ dz) / (dr @ dr)
    post_latent_directions[drug] = direction.astype(np.float32)
    print(drug, "latent direction norm:", float(np.linalg.norm(direction)),
          "factual innovation SD:", float(post_innov[active, dj].std()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 58–59: posterior dose-contrast feasibility\n"
            "- First fit failed on nonfinite reconstructed boundary contexts; excluded invalid rows without imputation.\n"
            "- Re-ran two-fold-by-stay decoder fits and saved posterior_contrast_cv.csv.\n"
            "- Computed original latent dose directions from counterfactual contrasts.\n"
            "- No weight edits; nine forward passes used.\n")
```
