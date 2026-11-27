The inventory check passed. `batch1` is exactly the stay set of the states in forward pass 2, so states and readings can be paired. The panels `P` and `PT` are indexed by stay and integer hour `ih`, and the dose at `ih` starts at ih+5 min. Readings are not hourly for every code; they have gaps.

Now the key feasibility test for a head-level correction. Can the glucose head's input linearly decode the insulin-dose residual, meaning the dose minus what the policy would have given? The head reads the view, which is built from the state after the bundle just before the reading (usually the drug bundle at t−1+5 min).

Steps:
1. Refit clean policy models on the training panel.
2. Compute dose residuals for the tuning stays.
3. Ridge-decode the residuals at several lags from a view proxy, `out + out_norm(mean_slots(h decayed over dt))`, with folds split by stay.

Interpretation:
- High cross-validated R² for the residuals at lags 1 to 6 means the head approach is viable.
- R² near zero means the view does not carry dose residuals, and the head approach would have to be dropped.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
def feats(D):
    g = D.groupby(level=0); o = pd.DataFrame(index=D.index)
    o['ih'] = D.index.get_level_values(1)
    for c, nm_ in [('glucose_finger','gf'),('glucose_lab','gl'),('map_arterial','ma'),('map_cuff','mc')]:
        o[nm_] = D[c].values; o[nm_+'_last'] = g[c].ffill().values
    gg = o.groupby(level=0)
    for nm_ in ['gf','gl','ma','mc']: o[nm_+'_plast'] = gg[nm_+'_last'].shift(1).values
    o['gf_plast2'] = gg['gf_last'].shift(2).values; o['ma_plast2'] = gg['ma_last'].shift(2).values
    for c, nm_ in [('insulin','ins'),('norepinephrine','ne')]:
        o[nm_+'_p1'] = g[c].shift(1).values; o[nm_+'_p2'] = g[c].shift(2).values
    return o
FP, FT = feats(P[['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']]), feats(PT[['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']])
pcols = [c for c in FP.columns]
mtr = FP['ih'].values >= 0; mtu = FT['ih'].values >= 0
pol = {}
for drug in ['insulin','norepinephrine']:
    m_ = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63)
    m_.fit(FP.loc[mtr, pcols], P.loc[mtr, drug]); pol[drug] = m_
    yt = PT.loc[mtu, drug].values; ph = m_.predict(FT.loc[mtu, pcols])
    print(drug, 'tuning R2', 1 - np.var(yt-ph)/np.var(yt), 'resid sd', np.std(yt-ph))
EPS = pd.DataFrame(index=PT.index); EPS['ins_hat'] = np.nan; EPS['ne_hat'] = np.nan
EPS.loc[mtu,'ins_hat'] = pol['insulin'].predict(FT.loc[mtu, pcols]); EPS.loc[mtu,'ne_hat'] = pol['norepinephrine'].predict(FT.loc[mtu, pcols])
EPS['e_ins'] = PT['insulin'] - EPS['ins_hat']; EPS['e_ne'] = PT['norepinephrine'] - EPS['ne_hat']
# states -> view proxy for each finger reading
x2, sid2, hr2 = S2['x'], S2['stay_id'], S2['hour'].astype(float)
fa = np.exp(load('core.flow.a').numpy().astype(np.float64)); onw, onb = load('core.out_norm.weight').numpy(), load('core.out_norm.bias').numpy()
def LN(v, w_, b_, eps=1e-5):
    mu = v.mean(-1, keepdims=True); sd = np.sqrt(v.var(-1, keepdims=True) + eps); return (v-mu)/sd*w_ + b_
recs = []
for s_id in batch1:
    ii = np.where(sid2 == s_id)[0]; hh_ = hr2[ii]
    for t_ in range(1, 25):
        if (s_id, t_) not in PT.index or np.isnan(PT.loc[(s_id, t_), 'glucose_finger']): continue
        j = ii[np.where(hh_ < t_ - 1e-6)[0][-1]]; dt_ = (t_ - hr2[j]) * 3600
        hmean = x2[j, 192:960].reshape(4, 192).mean(0) * np.exp(-dt_ * fa)
        view = x2[j, :192] + LN(hmean, onw, onb)
        r_ = {'stay': s_id, 't': t_, 'G': PT.loc[(s_id, t_), 'glucose_finger'], 'bhr': hr2[j]}
        for L in [1, 2, 3, 4, 6]:
            r_[f'ei{L}'] = EPS['e_ins'].get((s_id, t_-L), np.nan); r_[f'en{L}'] = EPS['e_ne'].get((s_id, t_-L), np.nan)
            r_[f'Di{L}'] = PT['insulin'].get((s_id, t_-L), np.nan)
        recs.append((r_, view))
R1 = pd.DataFrame([a for a, _ in recs]); VW = np.stack([b for _, b in recs])
print(R1.shape, 'bundle-lag frac (t - bhr):', pd.Series(np.round(R1['t'] - R1['bhr'], 3)).value_counts().head(4).to_dict())
gkf = GroupKFold(5)
for tgt in ['ei1','ei2','ei3','ei4','ei6','Di1','en1','en2','en3']:
    ok_ = R1[tgt].notna().values
    pr = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 12)), VW[ok_], R1.loc[ok_, tgt].values, groups=R1.loc[ok_, 'stay'].values, cv=gkf)
    yv = R1.loc[ok_, tgt].values; print(tgt, 'n', ok_.sum(), 'CV R2 %.3f' % (1 - np.var(yv-pr)/np.var(yv)), 'sd %.3f' % np.std(yv))
```