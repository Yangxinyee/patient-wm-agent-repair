The cache contains paired intervened and factual inputs. Its factual inputs differ from the general cache by only about \(3\times10^{-5}\) RMS, so I’ll use each pair’s own difference. I can now fit family-specific dose decoders. The test remains whether factual noise falls while dose sensitivity stays consistent across multipliers.

```python
cases207 = {}
for key, (rows, xc, xf, dr) in canon_cases200.items():
    mismatch = float(np.sqrt(np.mean((xf-Xraw204[:, rows])**2)))
    assert mismatch < 1e-3, (key, mismatch)
    assert np.max(np.abs(xc[..., -1]-1)) < 1e-6
    assert np.max(np.abs(xf[..., -1]-1)) < 1e-6
    cases207[key] = (rows, np.asarray(xc, dtype=np.float64),
                     np.asarray(xc-xf, dtype=np.float64),
                     np.asarray(dr, dtype=np.float64))

directions207 = {}
for drug in ("insulin", "norepinephrine"):
    own = family204[drug]
    FF = Xraw204[:, fitmask179 & own].reshape(-1, 193)
    FV = Xraw204[:, valmask179 & own].reshape(-1, 193)
    dc, yc = [], []
    for factor in (0., .5, 1.5, 2., 3.):
        rows, xc, dx, dr = cases207[(drug, factor)]
        use = fitmask179[rows] & own[rows]
        dc.append(dx[:, use].reshape(-1, 193))
        yc.append(np.broadcast_to(dr[use], (8, use.sum())).reshape(-1))
    DD, yy = np.concatenate(dc), np.concatenate(yc)
    cf = FF.T @ FF / len(FF)
    cd = DD.T @ DD / len(DD)
    rhs = DD.T @ yy / len(DD)
    reg = np.diag(np.maximum(np.diag(cf), 1e-4))
    print("\n", drug, "family-specific")
    for lam in (.001, .003, .01, .03, .1, .3, 1.):
        w = solve(cd + lam*cf + 1e-7*reg, rhs,
                  assume_a="pos", check_finite=False)
        directions207[(drug, lam)] = w
        slopes = {}
        for factor in (0., .5, 1.5, 2., 3.):
            rows, xc, dx, dr = cases207[(drug, factor)]
            use = valmask179[rows] & own[rows]
            dd = dx[:, use] @ w
            rr = np.broadcast_to(dr[use], dd.shape)
            slopes[factor] = round(float(np.sum(dd*rr)/np.sum(rr*rr)), 3)
        print(lam, "held-out factual SD", round(float(np.std(FV@w)), 4),
              "mean", round(float(np.mean(FV@w)), 4), "slopes", slopes)

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 201–207 / verified raw-head geometry\n"
            "Recovered exact raw head inputs (hidden reconstruction RMS 2.9e-7). "
            "Sensor-family gate is nearly exact (+1 glucose, -1 MAP). "
            "Canonical cases are paired intervened/factual raw inputs; "
            "their factual cache differs by only 3e-5 RMS from X200. "
            "Using paired differences and fitting family-specific dose directions. "
            "No weights changed; conservative five-tensor edit retained.\n")
print("\nNo weights changed.")
```