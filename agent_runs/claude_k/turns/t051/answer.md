Fitting the 8 `last_numeric` inputs as a free linear function of [1, has_last, age, z_last] cut the error from 3.06 to 0.59 units. That is about the noise level of the model's 4 random z draws.

So the state → view → q part of my rebuild is right, and the only uncertain piece is the exact input layout of `last_numeric`. The fitted matrix agrees with my guess for the first four inputs (has_last, age, z_last, has_last). The other four seem to carry z_last again in some form. With 32 parameters fitted on 50 readings I won't read more into it.

That accuracy is enough for the next question: does `value_bin.0` have hidden units that never fire on real readings? Such units could be turned into a gated dose pathway without changing any current prediction.

This run rebuilds q and the head input for every reading in the 99 saved stays, using the fitted layout. It then reports, for each of the 192 hidden units, how often it is active, its largest pre-activation and the size of its output weights.

If there are units with zero activity and a clear negative margin, I can repurpose them. If none exist, I will need a different route, such as weak units whose removal costs almost nothing.

```python
stays99 = sorted(st.stay.unique())
R99 = tr[tr.stay_id.isin(stays99) & tr.event.isin(list(CEN))].sort_values(['stay_id', 'hour']).reset_index(drop=True)
R99['prev_val'] = R99.groupby(['stay_id', 'event']).value.shift(1)
R99['prev_hr'] = R99.groupby(['stay_id', 'event']).hour.shift(1)
st_s = st.sort_values(['stay', 'hour'])
kidx = np.empty(len(R99), int)
for s_, g_ in R99.groupby('stay_id'):
    ss = st_s[st_s.stay == s_]
    pos = np.searchsorted(ss.hour.values, g_.hour.values - 1e-9) - 1
    kidx[g_.index.values] = ss.index.values[pos]
R99['k'] = kidx; R99['dt'] = (R99.hour.values - st.hour.values[kidx]) * 3600
Xk = X[kidx].astype(np.float64)
out_ = Xk[:, :192]; hh = Xk[:, 192:960].reshape(-1, 4, 192)
hf_ = hh * np.exp(-R99.dt.values[:, None, None] * flow_k[None, None, :])
ctx_ = out_ + ln(hf_.mean(1), W['core.out_norm.weight'], W['core.out_norm.bias']) + mlp(np.log(R99.dt.values)[:, None] * 0.1, 'gap')
zm_ = mlp(ctx_, 'z_prior')[:, :32]
Q99 = ln(ctx_ + lin(zm_, 'z_head') + W['type_emb.weight'][1], W['query_norm.weight'], W['query_norm.bias'])
cc = R99.event.map(CODE).values
has_ = (~R99.prev_val.isna()).values.astype(float)
zl_ = np.array([zsc(e_, 'value', v_) if not np.isnan(v_) else 0.0 for e_, v_ in zip(R99.event, R99.prev_val)])
lb_ = np.array([fbin(e_, 'value', v_) if not np.isnan(v_) else 0 for e_, v_ in zip(R99.event, R99.prev_val)])
ag_ = np.where(has_ > 0, np.log1p((R99.hour - R99.prev_hr).fillna(0).values) / 10, 0.0)
G4 = np.stack([np.ones(len(R99)), has_, ag_, zl_], 1)
WC99 = Q99 + W['embed.code.weight'][cc] + W['embed.bins.value.weight'][lb_] + W['last_numeric.bias'] + (G4 @ A1.T) @ Wln.T
PRE = lin(WC99, 'attr_heads.value_bin.0')
act_frac = (PRE > 0).mean(0); pmax = PRE.max(0); w2n = np.linalg.norm(W['attr_heads.value_bin.2.weight'], axis=0)
print('n readings', len(R99))
order = np.argsort(act_frac)
print('units with lowest activity: (unit, frac active, max pre, |w2|)')
for j_ in order[:20]: print(j_, round(act_frac[j_], 4), round(pmax[j_], 2), round(w2n[j_], 3))
print('n units never active:', int((act_frac == 0).sum()), ' with max pre < -1:', int((pmax < -1).sum()))
```