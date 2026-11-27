Enforcing all multiplier responses exactly requires large factual MAP shifts, so those fits are unsuitable. I’ll test a more conservative route that prioritizes doubled and half doses, limits factual variance, and checks the other multipliers for failures. It must also leave the validated insulin responses largely intact.

```python
def rr_conservative_map():
    A, target = rr_NA, rr_Ntargets
    covariance = r257_Cm + 0.1*r257_Cw
    scale = np.trace(covariance)/192
    M = covariance + (1e-5*scale + 1e-8)*np.eye(192)
    subsets = {
        "N2_Nhalf": [5, 7],
        "N2_Nhalf_I2_Ihalf": [5, 7, 0, 2],
        "Nall_I2_Ihalf": [5, 6, 7, 8, 9, 0, 2],
    }
    fits = {}
    for label, selection in subsets.items():
        B, y = A[selection], target[selection]
        K = np.linalg.solve(M, B.T)
        v = K @ np.linalg.solve(B@K + 1e-5*np.eye(len(selection)), y)
        fact = r257_Fm @ v
        fits[label] = v
        print(label, "fact sd", round(float(fact.std()), 4),
              "norm", round(float(np.linalg.norm(v)), 2),
              "all slopes", np.round(A@v, 3).tolist())

    # Keep the insulin constraints; cap the factual feature energy.
    v = fits["N2_Nhalf_I2_Ihalf"]
    sd = float((r257_Fm@v).std())
    amplitude = min(1.0, 0.50/max(sd, 1e-12))
    v = v*amplitude
    correction = np.outer(rr_Ndirection/r245_delta, v)
    total = r255_CI + correction
    factual = r245_cases[("factual", 1.0)]
    old_fact = r256_predict(factual, r255_CI, 1.0)
    new_fact = r256_predict(factual, total, 1.0)
    print("\nSELECTED amplitude", amplitude,
          "factual feature sd", (r257_Fm@v).std(),
          "correction norm", np.linalg.norm(correction))
    print("factual local mean shift", (new_fact-old_fact).mean(0),
          "RMS", np.sqrt(((new_fact-old_fact)**2).mean(0)))
    shifts = {}
    for key in [("insulin", 2.0), ("insulin", 0.5),
                ("norepinephrine", 2.0), ("norepinephrine", 0.5),
                ("norepinephrine", 0.0), ("norepinephrine", 1.5),
                ("norepinephrine", 3.0)]:
        p = r256_predict(r245_cases[key], total, 1.0)
        shifts[key] = (p-new_fact).mean(0)
        print(key, "local shifts", np.round(shifts[key], 4).tolist())
    globals().update(rr_Nsubset_fits=fits, rr_Nv=v,
                     rr_NC=correction, rr_Ntotal=total,
                     rr_Nlocal_shifts=shifts)
    with open("/work/notes.md", "a") as f:
        f.write("\n## Run 262: conservative MAP route\n"
                "- Exact all-multiplier constraints produced unacceptable factual feature energy.\n"
                "- Fitted selected dose constraints and capped factual feature SD at 0.50 MAP units locally.\n"
                "- Checking insulin cross-talk and NE x0/.5/1.5/2/3 responses before submission.\n"
                "- No weights changed. Python runs: 262; forward passes: 21.\n")

rr_conservative_map()
```