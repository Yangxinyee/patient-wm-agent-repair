The LDA gate, built orthogonal to the z-noise subspace, separates glucose codes from MAP codes in the replica by 220 to 525 within-code standard deviations. Gating is therefore feasible. With two cancelling units it also tolerates noise far beyond the replica's.

Two further findings change the plan:
1. The stored decoders have about 38% of their norm in the z subspace. In the tool they would pick up z-draw noise, so I will refit them orthogonal to that subspace.
2. The evaluation reads only glucose_finger and map_arterial. Applying each correction only to finger rows (insulin) and arterial rows (NE) cuts the factual cost, since cuff (29% of rows) and lab (10%) rows are untouched.

This run computes:
- z-orthogonal decoders, fitted so that v·dX tracks the mean dose change over lags 1–3 on the replica edit responses, with a noise penalty from the within-cell covariance;
- per-code LDA gates (finger vs the rest, art vs the rest) at several regularisations;
- the four least-important hidden units to repurpose.

A usable decoder needs slope near 1 and noise per unit response no worse than the earlier values (about 1.2 for insulin, 0.2 for NE).

```python
from scipy.special import erf
gel_ = lambda a: 0.5 * a * (1 + erf(a / np.sqrt(2)))
print('VOCAB keys', list(VOCAB.keys())[:15])
print('dd mean|.| per col ins', np.abs(EDR[('insulin', 2.0)]['dd']).mean(0).round(3), 'ne', np.abs(EDR[('norepinephrine', 2.0)]['dd']).mean(0).round(4))
def cellres(idx):
    R = X64[idx].copy(); lb_o = lbf[idx]
    for b in np.unique(lb_o):
        m = lb_o == b; R[m] -= R[m].mean(0)
    return R
DECZ = {}
for drug, con in [('insulin', 0), ('norepinephrine', 2)]:
    gi = np.concatenate([EDR[(drug, f)]['gi'] for f in (2.0, 0.5)])
    dX = np.concatenate([EDR[(drug, f)]['dX'] for f in (2.0, 0.5)]).astype(np.float64)
    dd = np.concatenate([EDR[(drug, f)]['dd'] for f in (2.0, 0.5)]).astype(np.float64)
    re = np.concatenate([EDR[(drug, f)]['reached'] for f in (2.0, 0.5)]).astype(bool)
    sel = re & (cf[gi] == con)
    dXs = dX[sel] @ Pz; y = dd[sel, 0:3].mean(1)
    R = cellres(np.where(cf == con)[0]); S = Pz @ (R.T @ R / len(R)) @ Pz
    for kap in [0.003, 0.01, 0.03, 0.1, 0.3]:
        A = dXs.T @ dXs / len(y) + kap * S + 1e-9 * np.eye(192)
        v = Pz @ np.linalg.solve(A, dXs.T @ y / len(y))
        resp = dX[sel] @ v; slope = (resp @ y) / (y @ y); noise = np.sqrt(v @ S @ v)
        r2 = 1 - ((resp - slope * y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        DECZ[(drug, kap)] = v
        print(drug, 'n %d kap %g slope %.3f noise %.3f per-unit %.3f R2 %.3f' % (sel.sum(), kap, slope, noise, noise / slope, r2))
GATE = {}
for con in [0, 2]:
    on = cf == con
    Dg = X64[on].mean(0) - np.mean([X64[cf == c].mean(0) for c in range(4) if c != con], axis=0)
    for lamf in [1e-3, 1e-2, 3e-2, 1e-1]:
        lam = lamf * np.trace(Sw) / 192
        wg = Pz @ np.linalg.solve(Pz @ Sw @ Pz + lam * np.eye(192), Pz @ Dg)
        tg = X64 @ wg
        lo_on = np.percentile(tg[on], .01); hi_off = max(np.percentile(tg[cf == c], 99.99) for c in range(4) if c != con)
        sdw = max(tg[cf == c].std() for c in range(4))
        GATE[(con, lamf)] = (wg, lo_on, hi_off)
        print('gate code', con, 'lamf %g margin/sd %.1f  sd_on %.4f' % (lamf, (lo_on - hi_off) / sdw, tg[on].std() / (lo_on - hi_off)))
W1o = np.asarray(load_original('attr_heads.value_bin.0.weight'), np.float64); b1o = np.asarray(load_original('attr_heads.value_bin.0.bias'), np.float64)
W2o = np.asarray(load_original('attr_heads.value_bin.2.weight'), np.float64)
rs = np.random.default_rng(0).choice(len(X64), 20000, replace=False)
Hs = gel_(X64[rs] @ W1o.T + b1o)
key_ = cf[rs] * 17 + lbf[rs]
Hc = Hs.copy()
for kk in np.unique(key_):
    m = key_ == kk; Hc[m] -= Hc[m].mean(0)
imp = (Hc ** 2).mean(0) * (W2o ** 2).sum(0)
order = np.argsort(imp)
print('least important units', order[:8], 'imp', imp[order[:8]].round(5), 'median imp %.4f' % np.median(imp))
```