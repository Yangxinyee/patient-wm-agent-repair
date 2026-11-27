The original model’s insulin response does accumulate, but too weakly; doubling norepinephrine gives the wrong sign throughout. That argues against persistence alone as the repair. I’ll use paired factual predictions and states to estimate how the reading head responds to state changes. Good held-out prediction of those changes would support a selective head edit; poor prediction would refute the approximation.

```python
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

f9o = pd.read_csv("/work/fwd_9/factual_original.csv")
f9e = pd.read_csv("/work/fwd_9/factual_edited.csv")
f23e = pd.read_csv("/work/fwd_23/factual_edited.csv")
keys59 = ["stay_id", "hour", "event"]
assert f9o[keys59].equals(f9e[keys59])
assert f9e[keys59].equals(f23e[keys59])

state_lookup59 = {}
for sid in np.unique(state24_original["stay_id"]):
    ix = np.flatnonzero(state24_original["stay_id"] == sid)
    order = np.argsort(state24_original["hour"][ix], kind="stable")
    ix = ix[order]
    state_lookup59[int(sid)] = (state24_original["hour"][ix], ix)

previous59 = []
for row in f9o.itertuples(index=False):
    hh, ix = state_lookup59[int(row.stay_id)]
    pos = np.searchsorted(hh, row.hour - 1e-5, side="left") - 1
    previous59.append(int(ix[pos]) if pos >= 0 else -1)
previous59 = np.asarray(previous59)
valid59 = previous59 >= 0

def row_ln59(x):
    x = np.asarray(x, dtype=float)
    return (x-x.mean(axis=1, keepdims=True)) / np.sqrt(
        x.var(axis=1, keepdims=True)+1e-5)

proxy_original59 = row_ln59(state24_original["x"][previous59[valid59], :192])
proxy_edited59 = row_ln59(state24_edited["x"][previous59[valid59], :192])
proxy_delta59 = proxy_edited59 - proxy_original59
reading59 = f9o.loc[valid59].reset_index(drop=True)
numeric_dmean59 = (f9e.predicted_mean-f9o.predicted_mean).to_numpy()[valid59]
query_dmean59 = (f23e.predicted_mean-f9e.predicted_mean).to_numpy()[valid59]

empirical_head59 = {}
for event in ["glucose_finger", "map_arterial"]:
    mask = reading59.event.eq(event).to_numpy()
    groups59 = reading59.loc[mask, "stay_id"].to_numpy()
    experiments = {
        "numeric": (proxy_delta59[mask], numeric_dmean59[mask]),
        "query": (0.5*proxy_edited59[mask], query_dmean59[mask]),
    }
    empirical_head59[event] = {}
    print("\n", event, "n", int(mask.sum()))
    for experiment, (xx, yy) in experiments.items():
        results59 = []
        for penalty59 in [1e-5, 1e-3, 0.1, 10.0]:
            pp = np.zeros(len(yy))
            for tr, te in GroupKFold(4).split(xx, yy, groups59):
                reg = Ridge(alpha=penalty59, fit_intercept=False)
                reg.fit(xx[tr], yy[tr])
                pp[te] = reg.predict(xx[te])
            results59.append((float(np.mean((pp-yy)**2)), penalty59, r2_score(yy, pp)))
        mse59, penalty59, r259 = min(results59)
        reg = Ridge(alpha=penalty59, fit_intercept=False).fit(xx, yy)
        empirical_head59[event][experiment] = {
            "coef": reg.coef_.copy(), "penalty": penalty59,
            "rmse": np.sqrt(mse59), "r2": r259,
        }
        print(experiment, "target SD", round(float(yy.std()),4),
              "CV RMSE", round(float(np.sqrt(mse59)),4),
              "CV R2", round(float(r259),4), "ridge", penalty59)
    gn = empirical_head59[event]["numeric"]["coef"]
    gq = empirical_head59[event]["query"]["coef"]
    print("gradient cosine", round(float(gn@gq/(np.linalg.norm(gn)*np.linalg.norm(gq))),4),
          "numeric-gradient query R2",
          round(float(r2_score(experiments["query"][1], experiments["query"][0]@gn)),4),
          "query-gradient numeric R2",
          round(float(r2_score(experiments["numeric"][1], experiments["numeric"][0]@gq)),4))

print("\n=== NUMERIC-GAIN-4 TIME COURSE ===")
roll10 = pd.read_csv("/work/fwd_10/dose_edit_edited.csv")
print(roll10.loc[roll10.hour.between(13,24)].groupby(["hour","edit"])["shift"]
      .mean().unstack().round(3).to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 59: controlled head-response estimation\n"
            "- Aligned factual readings with preceding original/numeric-gain-4 states.\n"
            "- Fitting stay-held-out regressions to paired numeric and query-gain prediction changes.\n"
            "- These are empirical perturbation responses, not observational outcome associations.\n"
            "- No edits changed; forward passes used: 24.\n")
```