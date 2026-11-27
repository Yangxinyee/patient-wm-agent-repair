The states for the first 100 new stays have arrived (5,380 bundles).

Before requesting more, I want to know whether more data is still the limiting factor. If the probe's held-out R² stays near 0.42, the limit is how faintly the model encodes the dose, not sample size.

This run builds the probe dataset from q for every reading in all 199 stays I now have states for. The q values are rebuilt with the corrected GELU head code. It then fits stay-grouped ridge probes for these targets, comparing 99 against 199 stays:

- the random part of the insulin dose 1 and 2 hours before each glucose reading;
- the weighted target −1.5·ε₁ − 1.4·ε₂;
- the norepinephrine residual at lag 1 for MAP readings.

A clear rise in R² with more stays would mean more state passes are worth spending. A flat R² would mean I should stop collecting and design around the current probe quality.

```python
def build_set(Xs_, stay_, hour_):
    stdf = pd.DataFrame({'stay': stay_, 'hour': hour_}); stdf['row'] = np.arange(len(stdf))
    stl = stdf.sort_values(['stay', 'hour'])
    R_ = tr[tr.stay_id.isin(set(stay_.tolist())) & tr.event.isin(list(CEN))].sort_values(['stay_id', 'hour']).reset_index(drop=True)
    R_['prev_val'] = R_.groupby(['stay_id', 'event']).value.shift(1); R_['prev_hr'] = R_.groupby(['stay_id', 'event']).hour.shift(1)
    kk = np.empty(len(R_), int)
    for s_, g_ in R_.groupby('stay_id'):
        ss = stl[stl.stay == s_]; pos = np.searchsorted(ss.hour.values, g_.hour.values - 1e-9) - 1
        kk[g_.index.values] = ss.row.values[pos]
    dt_ = (R_.hour.values - hour_[kk]) * 3600
    Xk_ = Xs_[kk].astype(np.float64); o_ = Xk_[:, :192]; h_ = Xk_[:, 192:960].reshape(-1, 4, 192)
    hf__ = h_ * np.exp(-dt_[:, None, None] * flow_k[None, None, :])
    c_ = o_ + ln(hf__.mean(1), W['core.out_norm.weight'], W['core.out_norm.bias']) + mlp(np.log(dt_)[:, None] * 0.1, 'gap')
    Q_ = ln(c_ + lin(mlp(c_, 'z_prior')[:, :32], 'z_head') + W['type_emb.weight'][1], W['query_norm.weight'], W['query_norm.bias'])
    c__ = R_.event.map(CODE).values; has__ = (~R_.prev_val.isna()).values.astype(float)
    zl__ = np.array([zsc(e_, 'value', v_) if not np.isnan(v_) else 0.0 for e_, v_ in zip(R_.event, R_.prev_val)])
    lb__ = np.array([fbin(e_, 'value', v_) if not np.isnan(v_) else 0 for e_, v_ in zip(R_.event, R_.prev_val)])
    ag__ = np.where(has__ > 0, np.log1p((R_.hour - R_.prev_hr).fillna(0).values) / 10, 0.0)
    G4_ = np.stack([np.ones(len(R_)), has__, ag__, zl__], 1)
    base_ = Q_ + W['embed.code.weight'][c__] + W['embed.bins.value.weight'][lb__] + W['last_numeric.bias'] + (G4_ @ A2.T) @ Wln.T
    R_['ri'] = R_.stay_id.map(sidx).values; R_['th'] = R_.hour.values.astype(int)
    return dict(R=R_, Q=Q_, WCb=base_, code=c__, lb=lb__, kk=kk)
S5 = np.load('/work/fwd_5/states_original.npz')
D99 = build_set(X, st.stay.values, st.hour.values)
D5 = build_set(S5['x'], S5['stay_id'], S5['hour'])
def lags(D_, E_, L=3):
    ri_, th_ = D_['R'].ri.values, D_['R'].th.values
    return np.column_stack([np.where(th_ - l_ >= 0, E_[ri_, np.clip(th_ - l_, 0, 24)], 0.0) for l_ in range(1, L + 1)])
def probe_r2(Ds, evs_, E_, combo):
    Qs, ys, gs = [], [], []
    for D_ in Ds:
        m_ = D_['R'].event.isin(evs_).values & (D_['R'].th.values >= 1)
        Qs.append(D_['Q'][m_]); ys.append(lags(D_, E_)[m_] @ combo); gs.append(D_['R'].stay_id.values[m_])
    Q_, y_, g_ = np.vstack(Qs), np.concatenate(ys), np.concatenate(gs)
    yp = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), Q_, y_, groups=g_, cv=GroupKFold(5))
    return round(1 - np.mean((y_ - yp) ** 2) / np.var(y_), 3), len(y_)
GL = ['glucose_finger', 'glucose_lab']; MP = ['map_arterial', 'map_cuff']
for nm_, combo in [('ins eps1', [1, 0, 0]), ('ins eps2', [0, 1, 0]), ('ins -1.5e1-1.4e2', [-1.5, -1.4, 0])]:
    print(nm_, '99 stays', probe_r2([D99], GL, eps_ins, np.array(combo)), '199 stays', probe_r2([D99, D5], GL, eps_ins, np.array(combo)))
print('ne eps1', '99', probe_r2([D99], MP, eps_ne, np.array([1, 0, 0])), '199', probe_r2([D99, D5], MP, eps_ne, np.array([1, 0, 0])))
```