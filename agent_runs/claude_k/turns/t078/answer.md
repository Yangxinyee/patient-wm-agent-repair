The third batch of states arrived (5,359 bundles, 100 selected drug-heavy stays). This run:
- adds it to the probe set and re-checks the insulin probes on all 399 stays;
- prints the last 100 selected ids for the next request.

It also builds a better norepinephrine target. A gradient-boosting model, fitted on all 30,000 training stays, predicts each hour's norepinephrine rate from the MAP readings alone. Its residual keeps the carried-over part of earlier dose shocks, which is what a doubled dose actually changes. That residual should be both larger and better encoded in the state than the one-hour shock, whose probe R² is stuck near 0.14.

A clearly higher probe R² for this target would make the norepinephrine correction feasible. A low R² would mean that drug needs a different route.

```python
S7 = np.load('/work/fwd_7/states_original.npz'); D7 = build_set(S7['x'], S7['stay_id'], S7['hour'])
DS = [D99, D5, D6, D7]
for nm_, combo in [('ins eps1', [1, 0, 0]), ('ins -1.5e1-1.4e2', [-1.5, -1.4, 0])]:
    print(nm_, '399 stays', probe_r2(DS, GL, eps_ins, np.array(combo)))
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
mread = np.where(~np.isnan(ma), ma, mc)[:, :28]
feats_ne = []
for t_ in range(25):
    cols_ = [np.nan_to_num(mread[:, max(t_ - j_, 0)] if t_ - j_ >= 0 else np.full(30000, np.nan), nan=-1) for j_ in range(8)]
    past = mread[:, :t_ + 1]
    feats_ne.append(np.column_stack(cols_ + [np.nanmin(past, 1), np.nanmean(past, 1), np.nanmean(past < 65, 1), np.full(30000, t_)]))
FN = np.stack(feats_ne, 1)
Xne = FN.reshape(-1, FN.shape[2]); yne = ne_all.reshape(-1); grp_ne = np.repeat(np.arange(30000), 25)
tr_m = grp_ne % 2 == 0
gb1 = HGB(max_iter=300, learning_rate=0.08).fit(Xne[tr_m], yne[tr_m]); gb2 = HGB(max_iter=300, learning_rate=0.08).fit(Xne[~tr_m], yne[~tr_m])
pred_ne = np.where(tr_m, gb2.predict(Xne), gb1.predict(Xne)).reshape(30000, 25)
nu_ne = ne_all - pred_ne
print('readings-only R2 for norepi rate: %.3f' % (1 - np.var(nu_ne) / np.var(ne_all)), 'nu sd %.3f' % nu_ne.std())
print('ne nu1 399', probe_r2(DS, MP, nu_ne, np.array([1, 0, 0])), 'ne nu1+0.3nu2', probe_r2(DS, MP, nu_ne, np.array([1, 0.3, 0])))
rest2 = [s_ for s_ in new300 if s_ not in set(S5['stay_id'].tolist()) | set(S7['stay_id'].tolist())]
print(len(rest2)); print(json.dumps(rest2))
```