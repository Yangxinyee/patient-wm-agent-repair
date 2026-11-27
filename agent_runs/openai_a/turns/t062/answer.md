The insulin threshold estimates support starting with insulin alone. A decoder might still detect large off-policy doses even if it predicts small factual dose residuals poorly, but that needs testing. I’ll check its units and implied logit changes first; excessive factual shifts or weak dose information would argue against using it.

```python
print("=== FEATURE AND DECODER SCALING ===")
for label, features in (("train", HX), ("tuning", HTX)):
    print(label, "feature mean/sd/range",
          float(features.mean()), float(features.std()),
          float(features.min()), float(features.max()))
    for drug, dec in dose_decoders.items():
        raw_pred = features @ dec["w"] + dec["b"]
        standardized_pred = ((features - hidden_location) / hidden_scale) @ dec["w"] + dec["b"]
        print(drug, "raw prediction mean/sd", float(raw_pred.mean()), float(raw_pred.std()),
              "standardized prediction mean/sd",
              float(standardized_pred.mean()), float(standardized_pred.std()))

print("\n=== DOSE DATA CONTENTS ===")
for label, data, meta in (("train", hd_train, head_meta_train),
                          ("tuning", hd_tune, head_meta_tune)):
    print(label)
    for key in ("valid", "doses", "residual", "prior_hour"):
        val = np.asarray(data[key])
        print(key, val.shape, "first", val[:5].tolist())
    for event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
        use = meta.event.to_numpy() == event
        print(event, "n", int(use.sum()),
              "dose mean", np.nanmean(np.asarray(data["doses"])[use], axis=0).tolist(),
              "residual mean/sd",
              np.nanmean(np.asarray(data["residual"])[use], axis=0).tolist(),
              np.nanstd(np.asarray(data["residual"])[use], axis=0).tolist())

print("\n=== LOGIT-TILT SENSITIVITY ===")
centres_np = head_centres.cpu().numpy()
probs_np = np.asarray(head_probs_marginal)
rank_template = np.arange(17, dtype=np.float32) - 8.5
rank_template[0] = 0.0
pred_mean = np.sum(probs_np * centres_np, axis=1)
mean_rank = probs_np @ rank_template
tilt_sensitivity = np.sum(probs_np * centres_np * rank_template[None, :], axis=1) - pred_mean * mean_rank
for event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    use = head_meta_tune.event.to_numpy() == event
    print(event, "mean change per unit tilt, quantiles",
          np.quantile(tilt_sensitivity[use], [.1, .5, .9]).tolist())

print("\n=== ADDITIONAL STATE REQUEST IDS ===")
print(json.dumps({"type": "states", "model": "original", "split": "train",
                  "stays": [str(v) for v in train_ids[100:200]], "bundles": "all"}))

with open("/work/notes.md", "a") as f:
    f.write("\n## Decoder unit check; Python 56, forward 4\n"
            "- Insulin is the first repair target because threshold estimates support its negative effect.\n"
            "- Testing whether a residual decoder can respond off-policy without damaging factual fit; negative R2 remains a warning.\n"
            "- Norepinephrine is unchanged pending stronger identification and separate validation.\n")
```