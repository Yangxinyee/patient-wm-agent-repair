The action encoder represents dose non-monotonically, and infusion-END atoms use the measurement pathway without dose attributes. I’ll fit a small decoder for treatment innovations from the posterior’s hidden features, including zero targets for non-drug bundles. Held-out factual accuracy and approximately linear responses to changed doses would support using that decoder for a weight edit.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

torch.set_num_threads(4)
n4 = len(s4["hour"])
rows4 = np.arange(1, n4)
rows4 = rows4[s4["stay_id"][rows4] == s4["stay_id"][rows4 - 1]]
ctx4 = context_from_saved(rows4, s4)[0]
e4_used = wm_e4[rows4]

state_key4 = {(int(st), int(round(hr * 60))): j
              for j, (st, hr) in enumerate(zip(s4["stay_id"], s4["hour"]))}
acttable4 = records4[records4.event.isin(["insulin", "norepinephrine"])].copy()
acttable4["j"] = [state_key4[(int(st), int(round(hr * 60)))]
                  for st, hr in zip(acttable4.stay_id, acttable4.hour)]
counts_start4 = np.bincount(acttable4.j, minlength=n4).astype(np.float32)

def action_vectors4(drug=None, multiplier=1.0):
    aggregate = torch.zeros(n4, 192)
    for event in ["insulin", "norepinephrine"]:
        tab = acttable4[acttable4.event == event]
        mult = multiplier if event == drug else 1.0
        atoms = action_atom_exact(event, tab.rate.to_numpy() * mult,
                                 tab.dose.to_numpy() * mult)
        atom_hidden = mlp0(atoms, "encode_action.mlp1", "gelu")
        aggregate.index_add_(0, torch.as_tensor(tab.j.to_numpy(), dtype=torch.long),
                             atom_hidden)
    aggregate /= torch.from_numpy(np.sqrt(np.maximum(counts_start4, 1)))[:, None]
    size = torch.from_numpy(np.log1p(counts_start4))[:, None]
    return norm0(mlp0(aggregate + lin0(size, "encode_action.size"),
                      "encode_action.mlp2", "gelu"), "encode_action.norm")

def posterior_features4(action_vectors):
    return torch.nn.functional.gelu(
        lin0(torch.cat([ctx4, e4_used, action_vectors[rows4]], dim=1), "z_post.0")
    )

U4 = action_vectors4()
H4 = posterior_features4(U4).numpy()
zcalc4 = lin0(torch.from_numpy(H4), "z_post.2")[:, :32].numpy()
zerr4 = np.sqrt(np.mean((zcalc4 - s4["x"][rows4, 960:992]) ** 2, axis=1))
startmask4 = counts_start4[rows4] > 0
print("Posterior reconstruction RMSE, START/non-START:",
      np.mean(zerr4[startmask4]), np.mean(zerr4[~startmask4]))

R4 = np.zeros((len(rows4), 2))
dose_actual4 = np.zeros((n4, 2))
row_position4 = {int(j): i for i, j in enumerate(rows4)}
for k, drug in enumerate(["insulin", "norepinephrine"]):
    tab = acttable4[acttable4.event == drug]
    indices = pd.MultiIndex.from_arrays(
        [tab.stay_id.to_numpy(), np.floor(tab.hour.to_numpy()).astype(int)],
        names=panel.index.names)
    expected = nuisance_models[drug].predict(X.reindex(indices))
    for j, rate, mu in zip(tab.j, tab.rate, expected):
        dose_actual4[int(j), k] = rate
        if int(j) in row_position4:
            R4[row_position4[int(j)], k] = rate - mu

ids4_unique = np.unique(s4["stay_id"])
fit4 = np.isin(s4["stay_id"][rows4], ids4_unique[:50])
test4 = ~fit4
synthetic4 = []
for k, drug in enumerate(["insulin", "norepinephrine"]):
    for mult in [0.0, 0.5, 1.5, 2.0, 3.0]:
        Hcf = posterior_features4(action_vectors4(drug, mult)).numpy()
        Rcf = R4.copy()
        Rcf[:, k] += dose_actual4[rows4, k] * (mult - 1)
        affected = dose_actual4[rows4, k] > 0
        synthetic4.append((k, mult, Hcf, Rcf, affected))

HF = np.concatenate([H4[fit4]] +
                    [h[fit4 & affected] for k, m, h, r, affected in synthetic4])
RF = np.concatenate([R4[fit4]] +
                    [r[fit4 & affected] for k, m, h, r, affected in synthetic4])
decoder_trials4 = []
for alpha in [0.01, 0.1, 1., 10., 100.]:
    decoder = Ridge(alpha=alpha).fit(HF, RF)
    p = decoder.predict(H4[test4])
    delta_errors = []
    slopes = []
    for k, mult, hcf, rcf, affected in synthetic4:
        use = test4 & affected
        observed_delta = (decoder.predict(hcf[use]) - decoder.predict(H4[use]))[:, k]
        required_delta = (rcf[use] - R4[use])[:, k]
        delta_errors.append(np.mean((observed_delta - required_delta) ** 2))
        slopes.append(np.dot(observed_delta, required_delta) /
                      max(np.dot(required_delta, required_delta), 1e-12))
    decoder_trials4.append((alpha, decoder, np.mean(delta_errors)))
    print("alpha", alpha,
          "held-out factual R2", np.round(r2_score(R4[test4], p, multioutput="raw_values"), 3),
          "non-START RMS", np.round(np.sqrt(np.mean(p[~startmask4[test4]] ** 2, axis=0)), 4),
          "CF slope range", np.round([min(slopes), max(slopes)], 3),
          "CF delta MSE", round(np.mean(delta_errors), 5))
```
