The timing fix recovered 947 next-MAP matches, confirming that the empty audit was a matching problem. The training-record evidence supports a positive NE effect, but the previous edit failed factual fit. I’ll compare corrected linear and nonlinear detectors on held-out stays and rerun the residual diagnostic. A useful detector must respond to both dose edits while barely responding to factual records or insulin edits.

```python
import scipy.linalg as sla
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# Repaired residual diagnostic; this is not a new causal estimate.
audit100 = pd.DataFrame({
    "stay_id": groups96, "hour": cf_frame["hour"].to_numpy(),
    "row_index": np.arange(len(groups96)), "rate": dose_ne96
})
audit100 = pd.merge_asof(
    audit100.sort_values("hour"),
    maps99[["stay_id", "hour", "observed", "predicted_mean"]]
        .rename(columns={"hour": "map_hour"}).sort_values("map_hour"),
    left_on="hour", right_on="map_hour", by="stay_id",
    direction="forward", tolerance=1.05
).sort_values("row_index")
valid100 = audit100.predicted_mean.notna().to_numpy() & (dose_ne96 > 0)
ii100 = np.flatnonzero(valid100)
error100 = (audit100.observed - audit100.predicted_mean).to_numpy()
dose_pred100 = np.full(len(groups96), np.nan)
error_pred100 = np.full(len(groups96), np.nan)
for tr100, te100 in GroupKFold(5).split(Xctx96[ii100], groups=groups96[ii100]):
    train100, test100 = ii100[tr100], ii100[te100]
    for target100, output100 in ((dose_ne96, dose_pred100), (error100, error_pred100)):
        reg100 = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        reg100.fit(Xctx96[train100], target100[train100])
        output100[test100] = reg100.predict(Xctx96[test100])
dr100 = dose_ne96[ii100] - dose_pred100[ii100]
er100 = error100[ii100] - error_pred100[ii100]
beta100 = np.dot(dr100, er100) / np.dot(dr100, dr100)
scores100 = pd.Series(dr100 * (er100 - beta100 * dr100)).groupby(groups96[ii100]).sum()
se100 = np.sqrt(np.sum(scores100.to_numpy() ** 2)) / np.dot(dr100, dr100)
print("ORIGINAL NEXT-MAP RESIDUAL DIAGNOSTIC",
      {"n": len(ii100), "beta": float(beta100), "cluster_se": float(se100),
       "innovation_sd": float(dr100.std())})

# Recompute both detector frontiers in identical raw-rate target units.
frontier100 = []
detectors100 = {}
for route100, DF100, XF100, basis100, scale100 in (
    ("prelude_hidden", D94, XF94, null93, np.r_[scale93, 1.0]),
    ("b_in", D95, XF95, null95, scale95)
):
    train_fact100 = fit_fact_mask93
    test_fact100 = test_fact_mask93
    train_cf100 = fit_cf_mask93
    test_bundle100 = test_cf_mask93.reshape(-1, 4).all(axis=1)
    gram_fact100 = (XF100[train_fact100].T @ XF100[train_fact100]) / train_fact100.sum()
    grams100, rhs100 = {}, {}
    for key100, DX100 in DF100.items():
        yy100 = np.repeat((key100[1] - 1.0) * dose_ne96, 4)
        if key100[0] != "norepinephrine":
            yy100 = np.zeros_like(yy100)
        grams100[key100] = DX100[train_cf100].T @ DX100[train_cf100] / train_cf100.sum()
        rhs100[key100] = DX100[train_cf100].T @ yy100[train_cf100] / train_cf100.sum()
    for cross100, half100, anchor100 in (
        (30., 1., 300.), (30., 4., 300.), (30., 10., 300.),
        (30., 4., 3000.), (300., 4., 300.), (300., 4., 3000.)
    ):
        A100 = (grams100[("norepinephrine", 2.)]
                + half100 * grams100[("norepinephrine", .5)]
                + cross100 * (grams100[("insulin", 2.)] + grams100[("insulin", .5)])
                + anchor100 * gram_fact100)
        ridge100 = 1e-7 * np.trace(A100) / len(A100)
        theta100 = sla.solve(A100 + ridge100 * np.eye(len(A100)),
                            rhs100[("norepinephrine", 2.)]
                            + half100 * rhs100[("norepinephrine", .5)],
                            assume_a="pos")
        raw100 = (basis100 @ theta100) / scale100
        fp100 = XF100[test_fact100] @ theta100
        metric100 = {"route": route100, "cross": cross100, "half_w": half100,
                     "anchor": anchor100, "fact_rms": float(np.sqrt(np.mean(fp100**2)))}
        for key100, DX100 in DF100.items():
            pred100 = (DX100 @ theta100).reshape(-1, 4).mean(axis=1)
            select100 = test_bundle100 & (dose_ne96 > 0)
            if key100[0] == "norepinephrine":
                target100 = (key100[1] - 1.) * dose_ne96
                metric100[f"s{key100[1]}"] = float(
                    np.dot(target100[select100], pred100[select100]) /
                    np.dot(target100[select100], target100[select100]))
            else:
                metric100[f"ins_rms{key100[1]}"] = float(
                    np.sqrt(np.mean(pred100[test_bundle100] ** 2)))
        metric100["fact_per_x2"] = metric100["fact_rms"] / max(abs(metric100["s2.0"]), 1e-12)
        metric100["half_ratio"] = metric100["s0.5"] / max(metric100["s2.0"], 1e-12)
        key100 = (route100, cross100, half100, anchor100)
        detectors100[key100] = {"raw": raw100, "theta": theta100, "metrics": metric100}
        frontier100.append(metric100)

print("\nHELD-OUT DETECTOR FRONTIER")
print(pd.DataFrame(frontier100).round(5).to_string(index=False))
print("\nLOCAL CONTEXT OBJECTS")
for n100, o100 in list(globals().items()):
    if n100.startswith(("src45", "core_")) or (
        hasattr(o100, "shape") and tuple(o100.shape) in ((32,), (31,), (32, 192), (32, 4, 192))
    ):
        if hasattr(o100, "shape"):
            print(n100, tuple(o100.shape))
        elif isinstance(o100, dict):
            print(n100, "dict", list(o100)[:5])
        elif isinstance(o100, (tuple, list)):
            print(n100, str(o100)[:160])
print("W0 b_in: distance from current/original",
      float(np.linalg.norm(W0["core.b_in.weight"].cpu().numpy() - np.asarray(load("core.b_in.weight")))),
      float(np.linalg.norm(W0["core.b_in.weight"].cpu().numpy() - np.asarray(load_original("core.b_in.weight")))))

with open("/work/notes.md", "a") as f100:
    f100.write("\n## Run 100: corrected residual diagnostic and detector frontiers\n"
               f"Repaired next-MAP residual audit n={len(ii100)}, beta={beta100:.4f}, "
               f"cluster SE={se100:.4f}; diagnostic only, not independent causal identification. "
               "Refitted corrected linear and nonlinear NE detectors with held-out factual and insulin guards. "
               "No weights changed. Python runs: 100; forwards: 12.\n")
```