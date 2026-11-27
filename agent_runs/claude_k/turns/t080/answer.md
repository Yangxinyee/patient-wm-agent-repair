The model encodes norepinephrine roughly as a log-rate.

| Probe target | Held-out R² |
|---|---|
| Norepinephrine log-rate, 1 h before the reading | 0.86 |
| Norepinephrine raw rate, 1 h before | 0.63 |
| Norepinephrine log-rate minus its readings-only expectation | 0.35 (0.40 when the drug is running) |
| Insulin raw rate, 1 h before | 0.78 |

The real effect is linear in the rate, so a norepinephrine correction would have to turn a log-rate signal into a rate signal. That is possible with a few extra GELU units, but it is a second step.

I will build the insulin correction first, since the main readout depends on it. The first build steps are:
1. recompute which value-head hidden units never fire, now with the correct GELU head inputs on all 399 stays (about 20,000 readings);
2. design a gate direction in the head input that is strongly positive for the two glucose codes and strongly negative for the MAP codes, while staying nearly constant across readings.

Success means several units with a wide negative margin, and a gate whose worst-case glucose value is still far above the worst-case MAP value.

```python
DSall = [D99, D5, D6, D7]
WCall = np.vstack([D_['WCb'] for D_ in DSall]); codeall = np.concatenate([D_['code'] for D_ in DSall])
PREall = lin(WCall, 'attr_heads.value_bin.0')
pmax_ = PREall.max(0); act_ = (PREall > -1).mean(0)
dead = np.argsort(pmax_)[:16]
print('n readings', len(WCall)); print('most negative max pre-activations:', [(int(j_), round(pmax_[j_], 2)) for j_ in dead])
print('units with max pre < -3:', int((pmax_ < -3).sum()), ' < -2:', int((pmax_ < -2).sum()))
isG = np.isin(codeall, [6, 10]); isM = np.isin(codeall, [5, 7])
mu_ = WCall.mean(0); Cv = np.cov((WCall - np.where(isG[:, None], WCall[isG].mean(0), WCall[~isG].mean(0))).T)
dmean = WCall[isG].mean(0) - WCall[isM].mean(0)
ug = np.linalg.solve(Cv + 1e-3 * np.eye(192), dmean); ug /= np.linalg.norm(ug)
pg = WCall @ ug
print('gate proj: glucose min %.3f max %.3f | MAP min %.3f max %.3f' % (pg[isG].min(), pg[isG].max(), pg[isM].min(), pg[isM].max()))
for c_ in [5, 6, 7, 8, 9, 10, 11, 12]: print('code', c_, 'emb proj %.3f' % (W['embed.code.weight'][c_] @ ug))
```