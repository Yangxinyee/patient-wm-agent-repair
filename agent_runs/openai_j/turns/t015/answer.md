Sensor swapping changed insulin’s estimate markedly, as expected when doses respond to finger-reading noise. Norepinephrine’s proxy estimates are both positive, around 3–4, but not identical. I will now estimate fixed-dose response curves by accounting for later dose changes. Agreement with the insulin noise-instrument analysis would strengthen the targets; disagreement would require retaining wider uncertainty.

```python
from scipy.optimize import minimize_scalar

def estimate_response_kernel(dj, horizons=(1, 2, 3, 4, 6, 8)):
    s1, s2 = ((0, 1), (2, 3))[dj]
    moments = []
    grouped_y, grouped_a = [], []
    maxh = max(horizons)
    for h in horizons:
        for sj in (s1, s2):
            ts = np.arange(2, PT-h)
            valid = np.isfinite(PO[:, ts+h, sj])
            if dj == 0:
                valid &= np.isfinite(PO[:, ts, s1]) & np.isfinite(PO[:, ts, s2])
            si, ti = np.where(valid)
            tt = ts[ti]
            if dj == 0:
                z = PO[si, tt, s1] - PO[si, tt, s2]
                base = HPAST[si, tt]
                nonlinear = ((PF[si, tt-1]-pmed)/scales)**2
            else:
                z = innov[si, tt, dj]
                base = HX[si, tt]
                current = (PF[si, tt]-pmed)/scales
                nonlinear = np.column_stack([current**2, current**3,
                                              PA[si, tt-1]**2])
            cm = np.column_stack([np.ones(len(si)), base, nonlinear])
            zr = z - cm @ sla.lstsq(cm, z, lapack_driver="gelsy")[0]
            yy = PO[si, tt+h, sj]
            gy = np.bincount(si, weights=zr*yy, minlength=PN) / len(si)
            ga = np.zeros((maxh, PN))
            for j in range(h):
                ga[j] = np.bincount(
                    si, weights=zr*PA[si, tt+j, dj], minlength=PN
                ) / len(si)
            grouped_y.append(gy)
            grouped_a.append(ga)
            moments.append((h, sj, len(si)))

    gy = np.asarray(grouped_y)
    ga = np.asarray(grouped_a)
    ym = gy.sum(axis=1)
    am = ga.sum(axis=2)
    initial_beta = -2 if dj == 0 else 3.2
    powers0 = np.array([
        [0.9**(h-1-j) if j<h else 0 for j in range(maxh)]
        for h, _, _ in moments
    ])
    initial_xg = np.einsum("mj,mjn->mn", powers0, ga)
    initial_scores = gy - initial_beta * initial_xg
    variance = np.sum((initial_scores-initial_scores.mean(axis=1, keepdims=True))**2,
                      axis=1)
    weights = 1 / np.maximum(variance, 1e-15)

    def fit_moments(yvec, avec):
        def at_rho(rho):
            pw = np.array([
                [rho**(h-1-j) if j<h else 0 for j in range(maxh)]
                for h, _, _ in moments
            ])
            xv = (avec*pw).sum(axis=1)
            beta = np.dot(weights*xv, yvec) / np.dot(weights*xv, xv)
            loss = np.dot(weights, (yvec-beta*xv)**2)
            return loss, beta
        opt = minimize_scalar(lambda r: at_rho(r)[0], bounds=(0.1, 1.05),
                              method="bounded", options={"xatol": 1e-5})
        loss, beta = at_rho(opt.x)
        return float(opt.x), float(beta), float(loss)

    rho_k, beta_k, loss_k = fit_moments(ym, am)
    rng = np.random.default_rng(619+dj)
    boot = []
    for b in range(100):
        mult = rng.normal(0, 1, PN)
        by = ym + gy @ mult
        ba = am + np.einsum("mjn,n->mj", ga, mult)
        boot.append(fit_moments(by, ba)[:2])
    boot = np.asarray(boot)
    print("\n", DS[dj], "kernel rho/beta:", rho_k, beta_k,
          "bootstrap 10/50/90 percentiles:", np.quantile(boot, [.1, .5, .9], axis=0))
    diagnostics = []
    for m, (h, sj, n) in enumerate(moments):
        pw = np.array([rho_k**(h-1-j) if j<h else 0 for j in range(maxh)])
        denom = np.dot(am[m], pw)
        diagnostics.append(dict(horizon=h, sensor=MS[sj], n=n,
                                implied_beta=ym[m]/denom))
    print(pd.DataFrame(diagnostics).round(4).to_string(index=False))
    return dict(rho=rho_k, beta=beta_k, boot=boot,
                diagnostics=pd.DataFrame(diagnostics), moments=(gy, ga))

response_kernels = {d: estimate_response_kernel(j) for j, d in enumerate(DS)}

def kernel_readout(doses, hours, rho, beta, factor=2., start=12):
    change = np.zeros(doses.shape[0])
    reached = np.zeros(doses.shape[0], dtype=bool)
    total = np.zeros(doses.shape[0])
    counts = np.zeros(doses.shape[0])
    for t in range(len(hours)-1):
        if hours[t] >= start:
            change = rho*change + beta*(factor-1)*doses[:, t]
            reached |= doses[:, t] > 0
        else:
            change[:] = 0
        if hours[t+1] > start and hours[t+1] <= 24:
            total[reached] += change[reached]
            counts[reached] += 1
    keep = counts > 0
    return np.mean(total[keep]/counts[keep])

TPA = tuning_panel[DS].fillna(0).to_numpy().reshape(len(tuning_ids), PT, 2)
target_rows = []
for dj, d in enumerate(DS):
    rk = response_kernels[d]
    for fac in (0., .5, 1.5, 2., 3.):
        target_rows.append(dict(drug=d, factor=fac,
            first100=kernel_readout(TPA[:100, :, dj], phours, rk["rho"], rk["beta"], fac),
            all400=kernel_readout(TPA[:, :, dj], phours, rk["rho"], rk["beta"], fac)))
target_table = pd.DataFrame(target_rows)
print("\n=== RECORD-DERIVED FIXED-DOSE TARGETS; STILL ASSUMPTION-DEPENDENT ===")
print(target_table.round(4).to_string(index=False))
target_table.to_csv("/work/kernel_targets.csv", index=False)

with open("/work/notes.md", "a") as f:
    f.write("\n## Fixed-dose response-kernel estimates\n")
    f.write("- Fit geometric kernels to instrument/outcome moments, including subsequent recorded doses.\n")
    f.write("- Insulin instrument: paired-sensor noise difference with past-only controls.\n")
    f.write("- NE instrument: cross-fitted policy innovation with measured-history controls; stronger sequential-unconfoundedness assumption.\n")
    for d, rk in response_kernels.items():
        f.write(f"- {d}: rho={rk['rho']:.5f}, beta={rk['beta']:.5f}\n")
    f.write(target_table.to_string(index=False) + "\n")
    f.write("- Python runs: 12. Forward passes: 3. No edits yet.\n")
```
