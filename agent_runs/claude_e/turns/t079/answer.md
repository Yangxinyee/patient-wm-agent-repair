The head reconstruction is saved, so I can now evaluate value-head edits offline.

The edit I am aiming for adds, for glucose readings, the part of the insulin effect the model misses: γ·(dose − protocol-expected dose) at recent lags, with γ = β_true − β_model. NE works the same way on MAP.

Before spending forward passes on states, I will rebuild the true one-step regression on the training grid. That is gf at t+1 on tier-specific polynomials of gf at t, gf at t−1 and lab at t, plus dose-deviation lags 1–6. I will keep it as a reusable function so I can apply the identical specification to the model's predictions later. If it reproduces my earlier coefficients (about −2.0, −0.6, 0.1, 0.5, 0.6, 0.4), the pipeline is consistent.

```python
EDOSE = np.array([0, 1.03, 2.07, 3.24, 6.24]); EMISS = 0.03
def tier(g):
    t = np.full(g.shape, -1)
    t[g<150]=0; t[(g>=150)&(g<180)]=1; t[(g>=180)&(g<220)]=2; t[(g>=220)&(g<300)]=3; t[g>=300]=4
    t[np.isnan(g)] = -1
    return t
def build_ins_design(G, stays, nlag=6):
    gf = G['gf'].loc[stays].values; gl = G['gl'].loc[stays].values; ins = G['ins'].loc[stays].fillna(0).values
    hours = np.array(G['gf'].columns)
    tr = tier(gf); ed = np.where(tr>=0, EDOSE[np.clip(tr,0,4)], EMISS); dev = ins - ed
    rows = []
    H = len(hours)
    for i in range(1, H-1):          # predictor hour index i (hour t), outcome i+1
        y = gf[:, i+1]; g0 = gf[:, i]; ok = ~np.isnan(y) & ~np.isnan(g0)
        if i < nlag: continue
        for s in np.where(ok)[0]:
            rows.append((s, hours[i+1], y[s], g0[s], tr[s,i], gf[s,i-1], gl[s,i], *[dev[s,i-j] for j in range(nlag)], *[ins[s,i-j] for j in range(nlag)]))
    cols = ['s','hour','y','g0','tier','g1','lab'] + [f'dev{j+1}' for j in range(nlag)] + [f'ins{j+1}' for j in range(nlag)]
    return pd.DataFrame(rows, columns=cols)
def ins_regress(D, ycol='y', nlag=6, use='dev'):
    Xc = []
    for k in range(5):
        m = (D.tier==k).astype(float).values; c = D.g0.values - [140,165,200,260,330][k]
        Xc += [m, m*c, m*c**2]
    g1 = D.g1.values; Xc += [np.nan_to_num(g1 - D.g0.values), np.isnan(g1).astype(float)]
    lab = D.lab.values; Xc += [np.nan_to_num(lab - D.g0.values), np.isnan(lab).astype(float)]
    Xd = [D[f'{use}{j+1}'].values for j in range(nlag)]
    Xm = np.column_stack(Xc + Xd); yv = D[ycol].values
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    res = yv - Xm@beta; s2 = res.var(); cov = s2*np.linalg.pinv(Xm.T@Xm)
    nd = len(Xd)
    return beta[-nd:], np.sqrt(np.diag(cov))[-nd:], res.std()
ins_stays_tr = [s for s in stays_tr if (Gtr['ins'].loc[s].fillna(0)>0).any()]
Dtr = build_ins_design(Gtr, ins_stays_tr)
b, se, rsd = ins_regress(Dtr)
print('rows', len(Dtr), ' TRUE dev coefs', b.round(2), ' se', se.round(2), ' resid sd', round(rsd,2))
```