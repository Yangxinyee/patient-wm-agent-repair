The sensor-family gate is nearly exact, and the dose decoders recover substantial intervention signal. Their factual residuals are still noisy, especially for norepinephrine. I’ll refit each decoder using only its relevant sensors and check how the saved logit targets are defined. Lower residual noise with preserved dose slopes would justify adding gated features.

```python
from scipy.linalg import solve
from scipy.special import softmax

print("TARGET CONVENTION:")
for name in ["L179", "Lorig179", "Yfact179", "C179"]:
    a = arr192(globals()[name])
    print(name, a.shape, "range", [float(a.min()), float(a.max())])
print("Yfact minus L179 RMS:",
      np.sqrt(np.mean((arr192(Yfact179) - arr192(L179))**2)))
print("target_logits83 names:", target_logits83.__code__.co_names)
print("target_logits83 constants:", target_logits83.__code__.co_consts)
for drug in ["insulin", "norepinephrine"]:
    base = cf_features81[(drug, 1.0)]
    case = cf_features81[(drug, 2.0)]
    target = arr192(targets179[(drug, 2.0)])
    r = case["rows"]
    print(drug, "target range", [float(target.min()), float(target.max())],
          "target/base logits RMS", np.sqrt(np.mean(
              (target - arr192(base["logits"]))**2)))
    for label, logits in [("base", base["logits"]),
                          ("case", case["logits"]), ("target", target)]:
        means = (softmax(arr192(logits), axis=-1) * arr192(C179)[r]).sum(-1).mean(0)
        relevant = np.char.startswith(np.asarray(events179[r], dtype=str),
                                     "glucose" if drug == "insulin" else "map")
        print(label, "relevant mean", float(means[relevant].mean()))

directions193 = {}
for j, drug in enumerate(["insulin", "norepinephrine"]):
    prefix = "glucose" if j == 0 else "map"
    relevantF = np.char.startswith(np.asarray(events179, dtype=str), prefix)
    F = XF192[:, fitmask179 & relevantF].reshape(-1, 193)
    CF = F.T @ F / len(F)
    relevantD = np.zeros(len(D192), dtype=bool)
    for key, sl in slices192.items():
        rows_case = cf_features81[key]["rows"]
        relevantD[sl] = np.tile(np.char.startswith(
            np.asarray(events179[rows_case], dtype=str), prefix), 8)
    use = maskD192 & relevantD
    CD = D192[use].T @ D192[use] / use.sum()
    RD = D192[use].T @ Y192[use, j] / use.sum()
    for lam in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        coef = solve(CD + lam * CF + 1e-10 * np.eye(193),
                     RD, assume_a="pos")
        directions193[(drug, lam)] = coef
        factual = XF192[:, ~fitmask179 & relevantF].reshape(-1, 193) @ coef
        slopes = {}
        for factor in [0.0, 0.5, 2.0, 3.0]:
            sl = slices192[(drug, factor)]
            keep = (~maskD192[sl]) & relevantD[sl]
            yt = Y192[sl, j][keep]
            yp = D192[sl][keep] @ coef
            slopes[factor] = round(float(yt @ yp / max(yt @ yt, 1e-12)), 3)
        print(drug, "lambda", lam,
              "held-out factual RMS", round(float(np.sqrt(np.mean(factual**2))), 4),
              "factor slopes", slopes)

print("All saved model weights unchanged.")
```