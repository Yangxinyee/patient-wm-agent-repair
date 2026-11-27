The two-feature statistical model has candidates within the local factual-loss budget; some even slightly improve tuning loss. That alone does not establish a better dose response. I’ll now measure their exact held-out responses and check that the correction remains smooth and monotone, rather than gaining accuracy through an unstable curve.

```python
from scipy.special import ndtr

def slope237(cfmean, factmean, drug, factor, event, subset):
    mc = wm_cf146[(drug, factor)]
    mask = (np.asarray(subset, bool) & np.asarray(events206 == event) &
            np.asarray(mc["active"], bool))
    dr = np.asarray(mc["delta_rate"], float)[mask]
    return np.sum(dr*(cfmean-factmean)[mask])/max(np.square(dr).sum(), 1e-12)

print("CURRENT EXACT LOCAL SLOPES")
for subset_name, subset in [("fit", fitall208), ("heldout", cvall208)]:
    print(subset_name, {
        f"{event}/{factor}": round(slope237(
            curve236[("norepinephrine", factor)]["pred"]["mean"],
            pred233["mean"], "norepinephrine", factor, event, subset), 4)
        for factor in [.5, 2.] for event in ["map_arterial", "map_cuff"]})

rows237 = []
qgrid237 = np.linspace(-3, 3, 301)
for key, gamma in candidates236.items():
    factmean, _ = curve_eval236("train", gamma)
    row = dict(key=str(key), gamma=np.round(gamma, 5).tolist())
    row.update(frontier236[
        (frontier236.ridge == key[0]) & (frontier236.weight == key[1]) &
        (frontier236.increase == key[2])
    ].iloc[0][["cv_dNLL", "tune_dNLL"]].to_dict())
    for factor in [.5, 2.]:
        cfmean, _ = curve_eval236(("norepinephrine", factor), gamma)
        for event, label in [("map_arterial", "A"), ("map_cuff", "C")]:
            for subset_name, subset in [("fit", fitall208), ("cv", cvall208)]:
                row[f"{subset_name}_{label}{factor}"] = slope237(
                    cfmean, factmean, "norepinephrine", factor, event, subset)
    derivative = np.full_like(qgrid237, 1+gamma[1])
    for threshold, coefficient in zip([.25, .60], gamma[2:]):
        t = 8*(qgrid237-threshold)
        derivative += coefficient*(ndtr(t)+t*np.exp(-t*t/2)/np.sqrt(2*np.pi))
    row["min_derivative"] = derivative.min()
    row["max_derivative"] = derivative.max()
    rows237.append(row)
frontier237 = pd.DataFrame(rows237)
print("\nEXACT NONLINEAR FRONTIER")
columns237 = ["key", "cv_dNLL", "tune_dNLL", "fit_A2.0", "cv_A2.0",
              "cv_A0.5", "cv_C2.0", "cv_C0.5", "min_derivative",
              "max_derivative", "gamma"]
print(frontier237.sort_values("cv_A2.0", ascending=False)[columns237]
      .head(12).to_string(index=False))
frontier237.to_csv("/work/map_nonlinear_frontier.csv", index=False)

print("\nTUNING STAY IDS")
ids237 = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique()
print("count:", len(ids237), "first:", ids237[:8].tolist(),
      "last:", ids237[-8:].tolist())
print("cached tuning stays:", wm_tune_meta145["stay_id"].unique()[:8].tolist())

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 236–237: two-feature nonlinear MAP screen\n"
            "- Fit a four-coefficient statistical model (constant, existing linear "
            "signal, two smooth hinges), not the world-model weights.\n"
            "- 23 candidates passed preliminary held-out/tuning factual limits.\n"
            "- Checking exact counterfactual responses and monotonicity before "
            "allocating model neurons; no weights changed.\n")
```