The MAP feature is already centered on factual inputs, so a constant offset will not fix it. Its half-dose contrast is much weaker than its doubled-dose contrast, explaining the asymmetry. I’ll fit a replacement with explicit constraints across all five tested multipliers, then check its factual variance and exact local responses before submission.

```python
def rr_fit_map_features():
    cases = list(r245_cases)
    factual = r245_cases[("factual", 1.0)]
    base_mean = r256_predict(factual, r255_CI, 1.0)
    print("sensor order:", globals().get("r49_sensors"))

    # Measure downstream directions numerically, without gradients.
    directions = np.asarray(r245_dirs, dtype=np.float64)
    responses = []
    for j in range(2):
        d = torch.as_tensor(directions[:, j], dtype=torch.float32)
        plus = np242(r244_read_pre(factual["pre"] + 0.25*d, indices151))
        minus = np242(r244_read_pre(factual["pre"] - 0.25*d, indices151))
        responses.append(((plus-minus)/0.5).mean(0))
    responses = np.asarray(responses).T
    direction = directions @ np.linalg.solve(responses[[0, 2], :], [0.0, 1.0])
    print("direction responses (four sensors):",
          responses @ np.linalg.solve(responses[[0, 2], :], [0.0, 1.0]))
    print("direction norm:", np.linalg.norm(direction))

    E0 = np242(factual["E"]).mean(1)
    rows, targets, labels = [], [], []
    base_shifts = {}
    nonfactual = [k for k in cases if k[0] != "factual"]
    for j, key in enumerate(nonfactual):
        drug, factor = key
        delta_rate = r257_Ym[j*320:(j+1)*320].mean(0)
        denom = delta_rate[0 if drug == "insulin" else 1]
        current = r256_predict(r245_cases[key], r255_CI, 1.0)
        shift = (current-base_mean).mean(0)
        base_shifts[key] = shift
        rows.append((np242(r245_cases[key]["E"]).mean(1)-E0).mean(0)/denom)
        targets.append((2.95159917*delta_rate[1]-shift[2])/denom)
        labels.append(key)
    A = np.asarray(rows)
    target = np.asarray(targets)
    print("desired added MAP slope:", np.round(target, 4))
    print("current local shifts I2/Ihalf/N2/Nhalf:",
          {k: np.round(base_shifts[k], 4).tolist()
           for k in [("insulin", 2.0), ("insulin", 0.5),
                     ("norepinephrine", 2.0), ("norepinephrine", 0.5)]})

    fits = {}
    covariance = r257_Cm + 0.1*r257_Cw
    scale = np.trace(covariance)/192
    print("covariance scale:", scale)
    for ridge in [1e-5, 1e-4, 1e-3]:
        M = covariance + (ridge*scale + 1e-8)*np.eye(192)
        inverse_rows = np.linalg.solve(M, A.T)
        gram = A @ inverse_rows
        for softness in [1e-4, 1e-3, 1e-2, 1e-1]:
            v = inverse_rows @ np.linalg.solve(gram + softness*np.eye(len(target)), target)
            prediction = A @ v
            fact = r257_Fm @ v
            fits[(ridge, softness)] = v
            print("ridge/soft", ridge, softness,
                  "fact MAP-feature sd", round(float(fact.std()), 4),
                  "norm", round(float(np.linalg.norm(v)), 2),
                  "slopes", np.round(prediction, 3).tolist())

    globals().update(rr_Nfits=fits, rr_Ndirection=direction, rr_NA=A,
                     rr_Ntargets=target, rr_Nlabels=labels,
                     rr_Nbase_shifts=base_shifts)
    with open("/work/notes.md", "a") as f:
        f.write("\n## Run 261: constrained MAP dose-feature fits\n"
                "- Current features are centered; constant-bias correction is not justified.\n"
                "- Fitting minimum-factual-energy MAP features with explicit multiplier constraints.\n"
                "- Targets use record-estimated MAP impulse effect, not medical knowledge.\n"
                "- No weights changed. Python runs: 261; forward passes: 21.\n")

rr_fit_map_features()
```