On exposed tuning hours, the stored linear-response estimates are about −11.9/+5.9 for doubled/halved insulin and +4.5/−2.25 for norepinephrine. The persistence trial moves insulin toward those targets but fails factual fit. I will fit a cross-validated correction for its factual offset; GPU loss and rollouts will determine whether to retain it.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

X275 = inputs274[0].detach().numpy().astype(np.float64)
XO275 = oldinputs274[0].detach().numpy().astype(np.float64)
ordinary275 = np.setdiff1d(np.arange(192), [15, 21, 61, 66])
A275 = np.asarray(load("attr_heads.value_bin.0.weight")).copy()
b275 = np.asarray(load("attr_heads.value_bin.0.bias")).copy()
pre275 = X275 @ A275.T + b275
oldpre275 = XO275 @ A275.T + b275
center_fits275 = {}

for sensor in ["glucose_finger", "glucose_lab"]:
    signal, control = sensor_neurons[sensor]
    ix = np.flatnonzero(rows274.event.to_numpy() == sensor)
    use = ordinary275[np.std(X275[ix][:, ordinary275], axis=0) > 1e-4]
    Z = X275[ix][:, use]
    y = ((pre275[ix, signal] - pre275[ix, control])
         - (oldpre275[ix, signal] - oldpre275[ix, control]))
    groups = rows274.stay_id.to_numpy()[ix]
    cv = {}
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        pred = np.zeros(len(ix))
        for tr, va in GroupKFold(5).split(Z, y, groups):
            sc = StandardScaler().fit(Z[tr])
            reg = Ridge(alpha=alpha).fit(sc.transform(Z[tr]), y[tr])
            pred[va] = reg.predict(sc.transform(Z[va]))
        cv[alpha] = float(np.mean((pred - y) ** 2))
    alpha = min(cv, key=cv.get)
    sc = StandardScaler().fit(Z)
    reg = Ridge(alpha=alpha).fit(sc.transform(Z), y)
    coef = reg.coef_ / sc.scale_
    intercept = float(reg.intercept_ - coef @ sc.mean_)
    A275[control, use] += coef.astype(np.float32)
    b275[control] += intercept
    center_fits275[sensor] = {
        "features": use, "coef": coef, "intercept": intercept,
        "alpha": alpha, "cv_mse": cv
    }
    print(sensor, {
        "offset_mean": float(y.mean()), "offset_sd": float(y.std()),
        "zero_correction_rmse": float(np.sqrt(np.mean(y*y))),
        "cv_rmse": {a: round(m**0.5, 6) for a, m in cv.items()},
        "chosen_alpha": alpha,
        "active_signal_quantiles": np.quantile(pre275[ix, signal], [.01, .5, .99]).tolist(),
        "max_abs_added_weight": float(np.max(np.abs(coef)))
    })

save_file({k: torch.as_tensor(load(k)).clone() for k in edited()},
          "/work/checkpoint_before_center275.safetensors")
save_edit("attr_heads.value_bin.0.weight", torch.from_numpy(A275))
save_edit("attr_heads.value_bin.0.bias", torch.from_numpy(b275))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 275: persistence-centering trial\n"
            "- Matched fresh tau=1500 s training states to pre-trial training states.\n"
            "- Fitted only the incremental insulin factual offset, with five-fold stay-group CV.\n"
            "- Added ordinary-feature corrections to glucose control neurons; private drug coefficients unchanged.\n"
            "- Saved checkpoint_before_center275.safetensors. Trial remains unaccepted pending GPU factual and rollout tests.\n")
print("Installed reversible centering correction; next test is paired GPU factual fit.")
```
