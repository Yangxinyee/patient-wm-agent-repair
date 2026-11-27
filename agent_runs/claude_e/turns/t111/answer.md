Norepinephrine is harder to read from the head than insulin:
- lag-1 NE deviation on arterial MAP rows: cross-validated R² 0.18, read-out gain 0.19;
- the same on cuff MAP rows: R² 0.14;
- lag-1 insulin deviation on glucose-finger rows: R² 0.44, gain 0.44.

That limits how large each correction can be at no factual cost. For a correction of size a per unit of dose, scaled to unit gain, the factual squared error changes by about (a²/R² − 2aΔβ)·Var(dev). So it is cost-free up to |a| ≈ 2|Δβ|R².
- **Insulin:** with Δβ about −1.5 that bound is about 1.3–1.5. Somewhat larger values cost a little.
- **NE:** the bound is near 1, so the MAP fix will have to be partial.

Before choosing coefficients I need the norepinephrine one-step coefficients, true and model. This run fits them with the same specification on arterial MAP readings:
- the true coefficients from all 30k train stays;
- the model's from its predicted means on two samples: the random batch (fwd_6) and all 500 stays.

The four selected batches are biased in the way found last run, so I show the random batch separately. I expect the true lag-1 coefficient near the impulse-response value of +3.4 mmHg per unit rate, and the model's well below it. If the model already has most of it, the NE gap lies in the dynamics rather than the one-step response.

```python
def ne_design(G, stays):
    ma_g = G['ma'].loc[stays]; mc_g = G['mc'].loc[stays]; dvn = DEVNE.reindex(stays) if set(stays) <= set(DEVNE.index) else None
    mapv_g = ma_g.where(ma_g.notna(), mc_g)
    out_ = []
    for h in range(1, 26):
        if h not in ma_g.columns: continue
        d = pd.DataFrame({'sid': stays, 'hour': h, 'y': ma_g[h].values, 'm0': ma_g[h-1].values, 'm1': ma_g[h-2].values if h-2 in ma_g.columns else np.nan,
                          'c0': mc_g[h-1].values, 'band': band_of(mapv_g[h-1].values.astype(float))})
        for k in range(1, 7):
            d[f'nd{k}'] = dvn[h-k].values if (dvn is not None and h-k in dvn.columns) else 0.0
        out_.append(d)
    D_ = pd.concat(out_, ignore_index=True)
    D_ = D_[D_.y.notna() & D_.m0.notna()].reset_index(drop=True)
    for k in range(1, 7): D_[f'nd{k}'] = D_[f'nd{k}'].fillna(0.0)
    return D_
def ne_reg(D, ycol):
    D = D[D[ycol].notna()]; n = len(D); cols, names = [], []
    m0 = D.m0.values.astype(float); b = D.band.values.astype(int)
    for k in range(4):
        m = (b == k).astype(float)
        if m.sum() < 30: continue
        c = np.median(m0[b == k]); cols += [m, m*(m0-c)/10, m*((m0-c)/10)**2]; names += [f'b{k}', f'b{k}_m0', f'b{k}_m0sq']
    for v in ['m1', 'c0']:
        x = D[v].values.astype(float); miss = np.isnan(x)
        cols += [np.where(miss, 0, x - m0)/10, miss.astype(float)]; names += [f'{v}-m0', f'{v}_miss']
    for k in range(1, 7): cols.append(D[f'nd{k}'].values.astype(float)); names.append(f'nd{k}')
    Xm = np.column_stack(cols); y = D[ycol].values.astype(float)
    bb, *_ = np.linalg.lstsq(Xm, y, rcond=None); r = y - Xm @ bb; s2 = r.var()*n/(n-Xm.shape[1])
    se = np.sqrt(np.diag(np.linalg.pinv(Xm.T @ Xm))*s2)
    return pd.Series(bb, names), pd.Series(se, names), np.sqrt(s2)
ndv = [f'nd{k}' for k in range(1, 7)]
Dne_tr = ne_design(Gtr, [int(s) for s in Gtr['ma'].index])
bt, se_t, rt = ne_reg(Dne_tr, 'y')
print('TRUE (train all) nd:', bt[ndv].round(2).values, 'se', se_t[ndv].round(2).values, 'resid sd', round(rt, 2), 'rows', len(Dne_tr))
mg_ma = RA_[RA_.event == 'map_arterial'][['stay','hour','pm','file']].rename(columns={'stay':'sid'}); mg_ma['hour'] = mg_ma.hour.round().astype(int)
for nm, fsel in [('random batch fwd_6', [1]), ('all 5 files', [0,1,2,3,4])]:
    sts = sorted(set(int(s) for s in RA_.stay[RA_.file.isin(fsel)]))
    D_ = ne_design(Gtr, sts).merge(mg_ma[mg_ma.file.isin(fsel)], on=['sid','hour'], how='left')
    D_ = D_[D_.pm.notna()]
    by, sy, _ = ne_reg(D_, 'y'); bm, sm, rmm = ne_reg(D_, 'pm')
    print(f'{nm}: rows {len(D_)}\n  truth nd: {by[ndv].round(2).values} se {sy[ndv].round(2).values}\n  model nd: {bm[ndv].round(2).values} se {sm[ndv].round(2).values}')
```