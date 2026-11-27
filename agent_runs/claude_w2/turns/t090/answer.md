My earlier exploration left its winners in the kernel: activation silu, bins counted with side='right', a 'log' gap feature, a guessed 8-column layout for `last_numeric`, and a head NLL still about 0.02 too high. That head work is only partly done.

An exact local re-implementation would need these details right in every layer, and my earlier z check was off by about 0.12 against a z scale of 0.17. So building one is a long road.

**A cheaper route.** I first check whether the state already holds enough dose memory for a head-side correction.
- The correction would sit in one repurposed hidden unit of `attr_heads.value_bin`. It would be gated on code: glucose codes for insulin, MAP codes for norepinephrine.
- Its constant offset would be cancelled per code through the code-specific `embed_transition.value_bin` rows.
- It would read a linear decoder of the dose history.

**The test.** On the fwd_3 states I fit cross-validated ridge decoders, using the state of the bundle just before each glucose_finger reading (out plus slot-mean h). The targets are:
- insulin rate at lags 1–8;
- the deviation from the policy-expected rate at lags 1–4;
- an exponentially weighted dose history.

**How to read it.**
- If lags 1–4 decode at R² of roughly 0.7 or more, a decoder-based correction with the right concave shape is feasible.
- If only lag 1 decodes, I need either the local re-implementation (to build decoders that respond correctly to counterfactual doses) or a calibrated lag-1 correction.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, GroupKFold
q4g = fa4[fa4.event == 'glucose_finger'].copy()
q4g['h'] = q4g['hour'].round().astype(int)
b1 = b0.copy()
for k_ in range(1, 5): b1[f'e{k_}'] = b1[f'd{k_}'] - b1[f'p{k_}']
ew = np.zeros(len(b1)); 
for k_ in range(1, 11): ew += (0.8 ** (k_ - 1)) * b1[f'd{k_}'].values
b1['ewma'] = ew
Q = q4g.merge(b1[['stay_id','h'] + [f'd{k_}' for k_ in range(1, 9)] + [f'e{k_}' for k_ in range(1, 5)] + ['ewma','G1']], on=['stay_id','h'], how='inner')
Q = Q[Q['prev'].notna()]
prev_ = Q['prev'].astype(int).values
feat = np.concatenate([xS[prev_, :192], xS[prev_, 192:960].reshape(-1, 4, 192).mean(1)], 1)
grp_ = Q['stay_id'].values
print('n readings', len(Q), 'stays', len(np.unique(grp_)), 'prev hour gap', np.round(np.median(Q['hour'].values - hS_[prev_]), 3))
rcv = RidgeCV(alphas=np.logspace(-1, 4, 12))
for tg_ in [f'd{k_}' for k_ in range(1, 9)] + [f'e{k_}' for k_ in range(1, 5)] + ['ewma']:
    yv2 = Q[tg_].values
    pr_ = cross_val_predict(rcv, feat, yv2, cv=GroupKFold(5), groups=grp_)
    r2_ = 1 - np.var(yv2 - pr_) / np.var(yv2)
    print(f'{tg_:5s} sd {yv2.std():.2f}  CV R2 {r2_:.3f}')
```