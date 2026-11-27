The reconstruction has little average bias, but noticeable per-reading errors. I’ll check how much comes from latent sampling rather than incorrect conventions. I’ll also test whether its features encode dose deviations from the recorded treatment policy; successful decoding on the second training sample would support a targeted correction.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

headA64 = head_features_layout(cacheA, head_layout_spec["layout"],
                              ageunit=head_layout_spec["ageunit"], nsamples=64, seed=913)
draw_means = as_np((headA64["logits"].softmax(-1) *
                   cacheA["centers"][None, :, :]).sum(-1))
dm64 = as_np(headA64["means"]) - cacheA["gpu_mean"]
dn64 = as_np(headA64["nll"]) - cacheA["gpu_nll"]
print("RECONSTRUCTION 64 draws: mean bias/MAE",
      np.round([dm64.mean(), np.abs(dm64).mean()], 5),
      "NLL bias/MAE", np.round([dn64.mean(), np.abs(dn64).mean()], 5))
print("Latent draw mean SD", draw_means.std(0).mean(),
      "8-vs-64 mean MAE",
      np.abs(as_np(headA0["means"]) - as_np(headA64["means"])).mean())
for late in [False, True]:
    keep = (cacheA["df"]["hour"].to_numpy() >= 1) == late
    print("hour>=1", late, "n", keep.sum(), "mean/NLL MAE",
          np.abs(dm64[keep]).mean(), np.abs(dn64[keep]).mean())
print("Head feature local variables", head_features_layout.__code__.co_varnames)

def correction_data(cache, head):
    df = cache["df"]
    t = np.rint(df["hour"].to_numpy()).astype(int)
    keep = (t >= 1) & (t <= 24)
    rows = np.flatnonzero(keep)
    idx = pd.MultiIndex.from_arrays(
        [df["stay_id"].to_numpy()[keep], t[keep] - 1],
        names=panel.index.names)
    cov = X.loc[idx]
    actual_doses = panel.loc[idx, ["insulin", "norepinephrine"]].fillna(0).to_numpy()
    expected_doses = np.column_stack([
        nuisance_models[d].predict(cov) for d in ["insulin", "norepinephrine"]])
    innovations = actual_doses - expected_doses
    events = df["event"].to_numpy()[keep]
    glucose = np.char.startswith(events.astype(str), "glucose")
    iqrs = np.array([norm_meta[(e, "value")]["iqr"] for e in events])
    desired = np.where(glucose, dyn_estimates["G"]["beta"] * innovations[:, 0],
                       dyn_estimates["M"]["beta"] * innovations[:, 1])
    return dict(rows=rows, idx=idx, innovations=innovations, desired=desired,
                target=desired / iqrs, events=events, glucose=glucose,
                groups=df["stay_id"].to_numpy()[keep],
                actual=df["value"].to_numpy()[keep],
                original_mean=as_np(head["means"])[keep],
                H=as_np(head["hidden"]).mean(0)[keep],
                Q=as_np(head["wc"]).mean(0)[keep])

corrTrainA = correction_data(cacheTrain, headTrain0)
corrTrainB = correction_data(cacheTrainB, headTrainB0)
print("Innovation SD A/B",
      corrTrainA["innovations"].std(0), corrTrainB["innovations"].std(0))
decode_trials = {}
for feature in ["H", "Q"]:
    ha, hb = corrTrainA[feature], corrTrainB[feature]
    center, scale = ha.mean(0), np.maximum(ha.std(0), 0.03)
    for alpha in [0.01, 1.0, 100.0]:
        reg = Ridge(alpha=alpha).fit((ha-center)/scale, corrTrainA["target"])
        yp = reg.predict((hb-center)/scale)
        decode_trials[(feature, alpha)] = (reg, center, scale)
        print("CORRECTION DECODING", feature, alpha,
              "held-out R2", round(r2_score(corrTrainB["target"], yp), 4),
              "target/pred SD", np.round([corrTrainB["target"].std(), yp.std()], 4))
for c, label in [(corrTrainA, "train A"), (corrTrainB, "train B")]:
    for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        keep = c["events"] == event
        bb, ss = clustered_ols(c["innovations"][keep],
                              (c["actual"]-c["original_mean"])[keep],
                              c["groups"][keep])[:2]
        print("FACTUAL ERROR VS INNOVATIONS", label, event,
              "coefficients", np.round(bb[1:], 4), "SE", np.round(ss, 4))
```
