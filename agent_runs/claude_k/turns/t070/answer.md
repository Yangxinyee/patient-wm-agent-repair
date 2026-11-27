Forward pass 4 returned the original model's factual predictions for all 5,521 readings in the 99 saved stays. Its value NLL is 1.520 per reading. Glucose_finger has mean absolute error 11.9 and map_arterial 3.9.

This run rebuilds the value head for all these readings with GELU (time flow on, type row 1). It then fits the `last_numeric` input layout as a linear function of [1, has_last, age, z_last] on half the stays and checks it on the other half.

If the held-out error comes close to the noise from the model's 4 random z draws (around 0.5–1 unit), the rebuild is good enough to:
- find units that never fire;
- set the gate margins;
- estimate the factual cost of an edit offline.

```python
fa4 = pd.read_csv('/work/fwd_4/factual_original.csv')
M4 = R99.merge(fa4, left_on=['stay_id', 'hour', 'event'], right_on=['stay_id', 'hour', 'event'], how='left')
print('merged', len(M4), 'missing', M4.predicted_mean.isna().sum(), 'value match', np.allclose(M4.value, M4.observed))
dtv = R99.dt.values
Xk = X[R99.k.values].astype(np.float64); out_ = Xk[:, :192]; hh = Xk[:, 192:960].reshape(-1, 4, 192)
hf_ = hh * np.exp(-dtv[:, None, None] * flow_k[None, None, :])
ctx_ = out_ + ln(hf_.mean(1), W['core.out_norm.weight'], W['core.out_norm.bias']) + mlp(np.log(dtv)[:, None] * 0.1, 'gap')
zp_ = mlp(ctx_, 'z_prior')
Q99 = ln(ctx_ + lin(zp_[:, :32], 'z_head') + W['type_emb.weight'][1], W['query_norm.weight'], W['query_norm.bias'])
BASE99 = Q99 + W['embed.code.weight'][cc] + W['embed.bins.value.weight'][lb_] + W['last_numeric.bias']
TR99 = W['embed_transition.value_bin.weight'][cc * 17 + lb_]
CENm = np.stack([CEN[{5: 'map_cuff', 6: 'glucose_finger', 7: 'map_arterial', 10: 'glucose_lab'}[c_]] for c_ in cc])
def head_means(A, idx=None, ret_logits=False):
    idx = np.arange(len(R99)) if idx is None else idx
    wc = BASE99[idx] + (G4[idx] @ A.T) @ Wln.T
    lg = mlp(wc, 'attr_heads.value_bin') + TR99[idx]
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    m_ = (p[:, 1:] * CENm[idx]).sum(1)
    return (m_, lg) if ret_logits else m_
pmall = M4.predicted_mean.values
A_B = np.zeros((8, 4)); A_B[0, 1] = 1; A_B[1, 2] = 1; A_B[2, 3] = 1; A_B[3, 1] = 1
mB = head_means(A_B); print('layout B: MAE %.3f' % np.mean(np.abs(mB - pmall)))
stays_arr = R99.stay_id.values; half = np.isin(stays_arr, stays99[::2])
rng_ = np.random.default_rng(0); fit_idx = rng_.choice(np.where(half)[0], 800, replace=False)
fitA = least_squares(lambda a_: head_means(a_.reshape(8, 4), fit_idx) - pmall[fit_idx], A_B.ravel(), max_nfev=400)
A2 = fitA.x.reshape(8, 4); mA2 = head_means(A2)
for nm_, msk in [('fit half', half), ('held-out half', ~half)]:
    d_ = mA2[msk] - pmall[msk]; print(nm_, 'MAE %.3f' % np.mean(np.abs(d_)), {e_: '%.2f' % np.mean(np.abs(d_[R99.event.values[msk] == e_])) for e_ in CEN})
print(np.round(A2, 2))
```