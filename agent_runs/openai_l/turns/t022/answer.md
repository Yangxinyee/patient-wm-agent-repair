The threshold estimates show insulin effects persisting for several hours and then declining. That could fit a second-order response rather than simple exponential decay. I’ll use the first three innovation estimates to predict later lags, then test those predictions independently; a poor match would rule out using that response for calibration.

```python
with threadpool_limits(limits=8):
    for drug19 in ["insulin", "norepinephrine"]:
        for lag19 in range(3, 8):
            crossfit17((drug19, lag19), future17(A[drug19], lag19))
    for sensor19 in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        for horizon19 in range(4, 9):
            crossfit17((sensor19, horizon19), future17(P[sensor19], horizon19))

kernels19, kernel_se19, ar2_predictions19 = {}, {}, {}
for drug19, sensors19 in [
    ("insulin", ["glucose_finger", "glucose_lab"]),
    ("norepinephrine", ["map_arterial", "map_cuff"])
]:
    d19 = dml_dose_resid[drug19]
    for sensor19 in sensors19:
        info19 = persistence17[sensor19]
        beta19 = info19["beta_anchor"]
        kk19 = [beta19, beta19 * info19["rho_h2"], info19["kernel_lag2"]]
        ss19 = [np.nan, abs(beta19) * info19["rho_h2_SE"], np.nan]
        a119 = info19["rho_h2"]
        a219 = info19["lag2_over_beta"] - a119 ** 2
        predicted19 = kk19.copy()
        for lag19 in range(3, 8):
            predicted19.append(a119 * predicted19[-1] + a219 * predicted19[-2])
            yy19 = future_res17[(sensor19, lag19 + 1)].copy()
            for j19 in range(1, lag19 + 1):
                yy19 -= kk19[lag19 - j19] * future_res17[(drug19, j19)]
            ok19 = np.isfinite(yy19)
            den19 = np.sum(d19[ok19] ** 2)
            slope19 = np.sum(d19[ok19] * yy19[ok19]) / den19
            err19 = yy19[ok19] - slope19 * d19[ok19]
            score19 = np.bincount(groups14[ok19], weights=d19[ok19] * err19,
                                  minlength=30000)
            kk19.append(float(slope19))
            ss19.append(float(np.sqrt(score19 @ score19) / den19))
        kernels19[sensor19] = kk19
        kernel_se19[sensor19] = ss19
        ar2_predictions19[sensor19] = predicted19
        print("\n", sensor19, "AR2 coefficients", round(a119, 4), round(a219, 4))
        print("Measured kernels:", np.round(kk19, 4))
        print("Conditional SE:  ", np.round(ss19, 4))
        print("AR2 predictions: ", np.round(predicted19, 4))

print("\n=== NON-TRANSFORMER TENSOR NAMES ===")
for name19 in NAMES:
    if not any(s in name19 for s in [
        "core.blocks", "core.prelude", "core.coda",
        "embed_transition", "encode.", "encode_action."
    ]):
        print(name19, SHAPES[name19])
print("\nNumeric table type:", type(VOCAB["numeric"]).__name__)
print("Numeric note:", VOCAB["numeric_note"])
print("Forward files:", {d: os.listdir("/work/" + d) for d in ["fwd_1", "fwd_2"]})

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 19\nIndependent dynamic RD shows delayed and declining insulin response, suggesting testing AR2 rather than forcing AR1. Higher-horizon innovation kernels and out-of-horizon AR2 predictions retained in kernels19 and ar2_predictions19. Measured kernels: " + repr(kernels19) + "\n")
```
