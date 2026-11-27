Changed doses produce substantial posterior-feature changes, and the drug-only posterior reconstruction is exact. That makes this pathway testable, but does not yet show the changes have the right causal meaning. I’ll fit dose-contrast decoders with held-out stays; good counterfactual accuracy and low factual activation would support a rank-one probe.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

post_H = post_h0.numpy().astype(np.float64)
post_sids = repair_sf.stay_id.to_numpy()
post_unique_sids = np.unique(post_sids)
post_fold = np.searchsorted(post_unique_sids, post_sids) % 2
post_train_sids = np.sort(train.stay_id.unique())
post_sid_index = np.searchsorted(post_train_sids, post_sids)
post_hour_index = np.clip(np.floor(repair_sf.hour.to_numpy()).astype(int) + 1, 0, 25)
post_innov = innov[post_sid_index, post_hour_index].copy()
post_innov[post_rates == 0] = 0

post_cv_rows = []
post_full_models = {}
post_predictions = {}
post_train_factors = [0.0, 0.5, 1.5, 2.0, 3.0]

def fit_post_contrast(drug, dj, mode, factual_weight, alpha, fitting_mask):
    active = (post_rates[:, dj] > 0) & fitting_mask
    target0 = np.zeros(repair_n) if mode == "zero" else post_innov[:, dj]
    xx = [post_H[fitting_mask]]
    yy = [target0[fitting_mask]]
    ww = [np.full(fitting_mask.sum(), factual_weight)]
    for factor in post_train_factors:
        xx.append(post_cf[(drug, factor)][active])
        yy.append(target0[active] + (factor - 1) * post_rates[active, dj])
        ww.append(np.ones(active.sum()))
    xx = np.vstack(xx)
    yy = np.concatenate(yy)
    ww = np.concatenate(ww)
    xm = xx.mean(0)
    xs = np.maximum(xx.std(0), 0.05)
    reg = Ridge(alpha=alpha).fit((xx - xm) / xs, yy, sample_weight=ww)
    coef = reg.coef_ / xs
    intercept = float(reg.intercept_ - coef @ xm)
    return coef, intercept

for dj, drug in enumerate(["insulin", "norepinephrine"]):
    active = post_rates[:, dj] > 0
    for mode in ["zero", "innovation"]:
        for factual_weight in [2.0, 10.0]:
            for alpha in [0.1, 10.0, 100.0]:
                key = (drug, mode, factual_weight, alpha)
                pf = np.zeros(repair_n)
                pc = {factor: np.zeros(repair_n) for factor in post_train_factors}
                for held_fold in [0, 1]:
                    fit_mask = post_fold != held_fold
                    test_mask = ~fit_mask
                    coef, intercept = fit_post_contrast(
                        drug, dj, mode, factual_weight, alpha, fit_mask)
                    pf[test_mask] = post_H[test_mask] @ coef + intercept
                    for factor in post_train_factors:
                        pc[factor][test_mask] = (
                            post_cf[(drug, factor)][test_mask] @ coef + intercept)
                row = dict(drug=drug, mode=mode, factual_weight=factual_weight,
                           alpha=alpha,
                           factual_rms=float(np.sqrt(np.mean(pf**2))),
                           active_rms=float(np.sqrt(np.mean(pf[active]**2))),
                           off_action_rms=float(np.sqrt(np.mean(pf[~active]**2))))
                for factor, label in [(0.0, "zero"), (0.5, "half"), (2.0, "double"), (3.0, "triple")]:
                    true_delta = (factor - 1) * post_rates[active, dj]
                    pred_delta = pc[factor][active] - pf[active]
                    row["r2_" + label] = r2_score(true_delta, pred_delta)
                    row["gain_" + label] = float(
                        pred_delta @ true_delta / (true_delta @ true_delta))
                post_cv_rows.append(row)
                post_predictions[key] = (pf, pc)
                post_full_models[key] = fit_post_contrast(
                    drug, dj, mode, factual_weight, alpha, np.ones(repair_n, dtype=bool))

post_cv_table = pd.DataFrame(post_cv_rows)
print(post_cv_table.round(4).to_string(index=False))
post_cv_table.to_csv("/work/posterior_contrast_cv.csv", index=False)

post_latent_directions = {}
for dj, drug in enumerate(["insulin", "norepinephrine"]):
    active = post_rates[:, dj] > 0
    dh = np.vstack([post_cf[(drug, f)][active] - post_H[active]
                    for f in post_train_factors])
    dr = np.concatenate([(f - 1) * post_rates[active, dj]
                         for f in post_train_factors])
    dz = dh @ wt("z_post.2.weight")[:32].numpy().T
    direction = (dr @ dz) / (dr @ dr)
    post_latent_directions[drug] = direction.astype(np.float32)
    print(drug, "latent dose-direction norm:", float(np.linalg.norm(direction)),
          "mean absolute factual innovation:", float(np.mean(np.abs(post_innov[active, dj]))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 58: dose-contrast decoder cross-validation\n"
            "- Fitted small ridge decoders to posterior features, using factors 0/.5/1.5/2/3 and two folds by stay.\n"
            "- Compared zero-factual and recorded-policy-innovation targets; saved factual activation and held-out contrast accuracy.\n"
            "- Computed original latent dose directions by least squares on counterfactual feature contrasts.\n"
            "- No weight edits; nine forward passes used.\n")
```
