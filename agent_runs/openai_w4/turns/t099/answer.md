The six-lag estimates suggest persistence: insulin’s response grows at the second lag before decaying, while norepinephrine’s decays sooner. These remain provisional causal estimates. I will test whether raw action inputs can encode dose innovations accurately and whether drug bundles can be distinguished cleanly. Poor held-out accuracy or weak dose sensitivity would rule out this posterior-edit design.

```python
import sys
from sklearn.linear_model import Ridge
torch.set_num_threads(8)

def capture_call87(fn, *args):
    captured = {}
    old_profile = sys.getprofile()
    def hook(frame, event, arg):
        if frame.f_code is fn.__code__ and event == "return":
            captured.update(frame.f_locals)
    sys.setprofile(hook)
    try:
        result = fn(*args)
    finally:
        sys.setprofile(old_profile)
    return result, captured

raw87 = zp_input82.numpy().astype(np.float64)
raw_cf87 = {}
for drug87 in ["insulin", "norepinephrine"]:
    for factor87 in [0.5, 2.0]:
        _, cap87 = capture_call87(intervention_bank74, drug87, factor87)
        uu87 = cap87["u_cf"]
        print("CF encoding", drug87, factor87, tuple(uu87.shape))
        rr87 = raw87.copy()
        rr87[:, 384:] = uu87[idx82].numpy()
        raw_cf87[(drug87, factor87)] = rr87

print("MASKS", {drug: np.unique(weights82[drug], return_counts=True)
                for drug in labels82})
raw_candidates87, raw_rows87 = {}, []
for drug87 in ["insulin", "norepinephrine"]:
    dm87 = repair_bank["has_" + drug87][idx82].astype(bool)
    y87 = np.asarray(labels82[drug87])
    labelled87 = dm87 & (np.asarray(weights82[drug87]) > 0) & np.isfinite(y87)
    tr87, te87 = fit82 & labelled87, test82 & labelled87
    feat87 = np.concatenate([raw87[:, :192], raw87[:, 384:]], axis=1)
    mu87 = feat87[tr87].mean(0)
    sd87 = feat87[tr87].std(0)
    sd87[sd87 < 0.03] = 1.0
    design87 = np.column_stack([(feat87 - mu87) / sd87, np.ones(len(feat87))])
    gram87 = design87[tr87].T @ design87[tr87] / tr87.sum()
    rhs87 = design87[tr87].T @ y87[tr87] / tr87.sum()
    cg87, cr87 = np.zeros_like(gram87), np.zeros_like(rhs87)
    differences87 = {}
    for factor87 in [0.5, 2.0]:
        cf87 = raw_cf87[(drug87, factor87)]
        ff87 = np.concatenate([cf87[:, :192], cf87[:, 384:]], axis=1)
        diff87 = np.column_stack([(ff87 - feat87) / sd87, np.zeros(len(feat87))])
        dr87 = cfdr82[(drug87, factor87)]
        use87 = fit82 & dm87 & (np.abs(dr87) > 1e-7)
        cg87 += diff87[use87].T @ diff87[use87] / use87.sum()
        cr87 += diff87[use87].T @ dr87[use87] / use87.sum()
        differences87[factor87] = (diff87, dr87)
    for alpha87 in [0.001, 0.01, 0.1, 1.0]:
        for strength87 in [0.0, 1.0, 10.0]:
            penalty87 = np.eye(design87.shape[1]) * alpha87
            penalty87[-1, -1] = 1e-8
            c87 = np.linalg.solve(gram87 + strength87 * cg87 + penalty87,
                                 rhs87 + strength87 * cr87)
            prediction87 = design87 @ c87
            result87 = {"drug": drug87, "alpha": alpha87, "strength": strength87,
                        "held_R2": r2_score(y87[te87], prediction87[te87]),
                        "held_RMSE": np.sqrt(np.mean((y87[te87] - prediction87[te87])**2))}
            for factor87, (diff87, dr87) in differences87.items():
                use87 = test82 & dm87 & (np.abs(dr87) > 1e-7)
                result87["slope_" + str(factor87)] = (
                    np.dot(diff87[use87] @ c87, dr87[use87]) /
                    np.dot(dr87[use87], dr87[use87]))
            raw_rows87.append(result87)
            unscaled87 = np.zeros(577)
            unscaled87[:192] = c87[:192] / sd87[:192]
            unscaled87[384:576] = c87[192:384] / sd87[192:384]
            unscaled87[-1] = c87[-1] - np.dot(mu87 / sd87, c87[:-1])
            raw_candidates87[(drug87, alpha87, strength87)] = unscaled87
    gate87 = Ridge(alpha=0.1).fit(raw87[fit82, 384:], dm87[fit82].astype(float))
    gv87 = gate87.predict(raw87[test82, 384:])
    print("Drug gate", drug87,
          "held accuracy", np.mean((gv87 > 0.5) == dm87[test82]),
          "active quantiles", np.quantile(gv87[dm87[test82]], [0, .05, .5, .95, 1]),
          "inactive quantiles", np.quantile(gv87[~dm87[test82]], [0, .05, .5, .95, 1]))
raw_table87 = pd.DataFrame(raw_rows87)
print(raw_table87.round(4).to_string(index=False))
note("Run 87: tested raw posterior-input dose-innovation decoders and action-presence gates with stay-separated validation. No weights saved. Six-lag insulin kernel has a delayed peak, unlike a single exponential.")
```
