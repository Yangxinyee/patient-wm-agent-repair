The corrected scaling exactly reproduces the earlier decoder diagnostics. I’ll now fit a fresh insulin-only correction on both training batches, using dose residuals in physical units, and apply it as a reversible rank-one head edit. The CPU check is preliminary; GPU factual fit and dose rollouts will decide whether it is useful.

```python
from sklearn.linear_model import Ridge
from scipy.special import softmax, logsumexp

wanted_extra = set(state_files[5]["stay_id"].astype(int).tolist())
extra_chunks = []
for chunk in pd.read_csv("/data/train.csv",
                         usecols=["stay_id", "hour", "event", "value", "rate"],
                         chunksize=200000):
    selected = chunk[chunk.stay_id.isin(wanted_extra)]
    if len(selected):
        extra_chunks.append(selected.copy())
records_extra = pd.concat(extra_chunks, ignore_index=True)
head_meta_extra = reading_metadata(records_extra, state_files[5])

feat_first = make_head_features(head_meta_train, state_files[4])
feat_extra = make_head_features(head_meta_extra, state_files[5])
feat_tuning = make_head_features(head_meta_tune, state_files[3])
print("Default-builder versus cached raw hidden RMSE:",
      float(np.sqrt(np.mean((feat_first[0] - raw_hidden_train) ** 2))))
print("Extra records/readings:", records_extra.shape, head_meta_extra.shape)

hd_extra = head_dose_data(head_meta_extra, P, train_ids, policy_predictions_panel)
centres_by_event = {
    event: centres_np[np.flatnonzero(head_meta_tune.event.to_numpy() == event)[0]]
    for event in head_meta_tune.event.unique()
}

def head_unit_target(meta0, logits0, dose_data0):
    probabilities = softmax(logits0, axis=1)
    centres0 = np.stack([centres_by_event[event] for event in meta0.event])
    m0 = np.sum(probabilities * centres0, axis=1)
    covariance = (np.sum(probabilities * centres0 * rank_template[None, :], axis=1)
                  - m0 * (probabilities @ rank_template))
    is_glucose = meta0.event.isin(["glucose_finger", "glucose_lab"]).to_numpy()
    eligible = is_glucose & np.asarray(dose_data0["valid"])
    target0 = np.zeros(len(meta0), dtype=np.float32)
    target0[eligible] = (np.asarray(dose_data0["residual"])[eligible, 0]
                        / np.maximum(covariance[eligible], 5.0))
    return target0, covariance, eligible

unit_first, sens_first, eligible_first = head_unit_target(head_meta_train, feat_first[1], hd_train)
unit_extra, sens_extra, eligible_extra = head_unit_target(head_meta_extra, feat_extra[1], hd_extra)
unit_tuning, sens_tuning, eligible_tuning = head_unit_target(head_meta_tune, feat_tuning[1], hd_tune)

pooled_hidden = np.concatenate([feat_first[0], feat_extra[0]])
pooled_target = np.concatenate([unit_first, unit_extra])
pooled_location = pooled_hidden.mean(axis=0)
pooled_scale = np.maximum(pooled_hidden.std(axis=0), 0.05)
Zfirst = (feat_first[0] - pooled_location) / pooled_scale
Zextra = (feat_extra[0] - pooled_location) / pooled_scale
Ztuning = (feat_tuning[0] - pooled_location) / pooled_scale

cv_scores = {}
for alpha0 in (10.0, 100.0, 1000.0, 10000.0):
    fit0 = Ridge(alpha=alpha0).fit(Zfirst, unit_first)
    pextra = fit0.predict(Zextra)
    mse0 = float(np.mean((pextra - unit_extra) ** 2))
    r20 = 1.0 - mse0 / max(float(np.var(unit_extra)), 1e-12)
    cross_rms = float(np.sqrt(np.mean(
        (pextra[~eligible_extra] * sens_extra[~eligible_extra]) ** 2)))
    cv_scores[alpha0] = mse0
    print("alpha", alpha0, "independent-train R2", round(r20, 4),
          "cross-event physical RMS", round(cross_rms, 4))

insulin_head_alpha = min(cv_scores, key=cv_scores.get)
insulin_head_fit = Ridge(alpha=insulin_head_alpha).fit(
    (pooled_hidden - pooled_location) / pooled_scale, pooled_target)
insulin_head_w = (insulin_head_fit.coef_ / pooled_scale).astype(np.float32)
insulin_head_b = float(insulin_head_fit.intercept_ - pooled_location @ insulin_head_w)
insulin_probe_beta = -2.0

def approximate_edit_cost(meta0, feature_tuple):
    H0, L0 = feature_tuple[:2]
    tilt0 = insulin_probe_beta * (H0 @ insulin_head_w + insulin_head_b)
    L1 = L0 + tilt0[:, None] * rank_template[None, :]
    obs0 = meta0.observed_bin.to_numpy().astype(int)
    rows0 = np.arange(len(meta0))
    nll0 = logsumexp(L0, axis=1) - L0[rows0, obs0]
    nll1 = logsumexp(L1, axis=1) - L1[rows0, obs0]
    print("approximate NLL change", float(np.mean(nll1 - nll0)),
          "tilt SD", float(tilt0.std()))
    for event in meta0.event.unique():
        use0 = meta0.event.to_numpy() == event
        print(event, "NLL change", float(np.mean((nll1 - nll0)[use0])))

print("\nChosen alpha:", insulin_head_alpha)
print("TRAIN CPU CHECK")
approximate_edit_cost(head_meta_extra, feat_extra)
print("TUNING CPU CHECK")
approximate_edit_cost(head_meta_tune, feat_tuning)

for name in list(edited()):
    drop_edit(name)
output_weight_name = "attr_heads.value_bin.2.weight"
output_bias_name = "attr_heads.value_bin.2.bias"
save_edit(output_weight_name,
          as_numpy(load_original(output_weight_name))
          + insulin_probe_beta * np.outer(rank_template, insulin_head_w))
save_edit(output_bias_name,
          as_numpy(load_original(output_bias_name))
          + insulin_probe_beta * rank_template * insulin_head_b)

print("EDITS:", edited())
print("NEXT FACTUAL REQUEST:")
print(json.dumps({"type": "factual", "model": "both", "split": "tuning",
                  "stays": [int(v) for v in tuning_ids[:100]]}))

with open("/work/notes.md", "a") as f:
    f.write("\n## First reversible insulin head probe; Python 58, forward 5/50\n"
            f"- Rank-one final value-head edit; beta={insulin_probe_beta}, ridge alpha={insulin_head_alpha}.\n"
            "- Statistical target: factual insulin dose residual divided by local mean/logit-tilt sensitivity; zero on other outcomes.\n"
            "- Fitted using 200 training stays, no model gradients or weight optimizer.\n"
            "- CPU checks are approximate; GPU factual NLL and intervention response are the acceptance tests.\n")
```