The raw inputs reconstruct the original GELU features accurately, and the transition offset is effectively independent of latent draws. The case-tuple convention did not match my assumption, so no decoder fit ran. I’ll verify which arrays are factual and intervened before fitting; a verified convention should give signed contrasts matching the physical dose changes.

```python
print("=== VERIFY CANONICAL CASE CONVENTION ===")
cases205 = {}
for key, value in canon_cases200.items():
    rows, aa, bb, dr = value
    factual = Xraw204[:, rows]
    ea = float(np.sqrt(np.mean((aa-factual)**2)))
    eb = float(np.sqrt(np.mean((bb-factual)**2)))
    if key[1] == 2.0:
        print(key, "A factual RMS", ea, "B factual RMS", eb,
              "A last-column range", (float(aa[..., -1].min()), float(aa[..., -1].max())),
              "B last-column range", (float(bb[..., -1].min()), float(bb[..., -1].max())),
              "dose-change quantiles", np.quantile(dr, [0, .5, 1]))
    if ea < 2e-5 and np.max(np.abs(bb[..., -1]-1)) < 2e-5:
        xc, dx = bb, bb-aa
    elif eb < 2e-5 and np.max(np.abs(aa[..., -1]-1)) < 2e-5:
        xc, dx = aa, aa-bb
    elif np.max(np.abs(bb[..., -1])) < 2e-5:
        xc, dx = aa, bb
    elif np.max(np.abs(aa[..., -1])) < 2e-5:
        xc, dx = bb, aa
    else:
        raise ValueError("Unverified feature convention: " + str(key))
    cases205[key] = (rows, xc.astype(np.float64), dx.astype(np.float64),
                     np.asarray(dr, dtype=np.float64))

directions205 = {}
for drug in ("insulin", "norepinephrine"):
    own = family204[drug]
    FF = Xraw204[:, fitmask179 & own].reshape(-1, 193)
    FV = Xraw204[:, valmask179 & own].reshape(-1, 193)
    dc, yc = [], []
    for factor in (0., .5, 1.5, 2., 3.):
        rows, xc, dx, dr = cases205[(drug, factor)]
        use = fitmask179[rows] & own[rows]
        dc.append(dx[:, use].reshape(-1, 193))
        yc.append(np.broadcast_to(dr[use], (8, use.sum())).reshape(-1))
    DD = np.concatenate(dc)
    yy = np.concatenate(yc)
    cf = FF.T @ FF / len(FF)
    cd = DD.T @ DD / len(DD)
    rhs = DD.T @ yy / len(DD)
    reg = np.diag(np.maximum(np.diag(cf), 1e-4))
    print("\n", drug, "family-specific")
    for lam in (.001, .003, .01, .03, .1, .3, 1.):
        w = solve(cd + lam*cf + 1e-7*reg, rhs,
                  assume_a="pos", check_finite=False)
        directions205[(drug, lam)] = w
        slopes = {}
        for factor in (0., .5, 1.5, 2., 3.):
            rows, xc, dx, dr = cases205[(drug, factor)]
            use = valmask179[rows] & own[rows]
            dd = dx[:, use] @ w
            rr = np.broadcast_to(dr[use], dd.shape)
            slopes[factor] = round(float(np.sum(dd*rr)/np.sum(rr*rr)), 3)
        print(lam, "held-out factual SD", round(float(np.std(FV@w)), 4),
              "mean", round(float(np.mean(FV@w)), 4), "slopes", slopes)
print("\nNo weights changed.")
```