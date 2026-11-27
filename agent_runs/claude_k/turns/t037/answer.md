The data's conditional kernel is not a simple decaying "insulin on board" kernel.

- **Insulin:** c_l = -2.01, -1.45, -0.47, then positive +0.22, +0.73, +1.09 at lags 3–5.
- **Norepinephrine:** +3.30, +0.99, -0.25, -0.99, -1.09, -1.45.

Some of the sign reversal is real structure. Glucose relaxes toward a patient baseline, so a model that holds the readings fixed must credit older doses for having pushed the readings down, otherwise it misjudges the baseline. Part is also confounding, because I controlled only 4 hours of readings. Either way, deriving the exact correction kernel analytically is unreliable.

The practical route is empirical. Add a dose-history correction to the value head for glucose (driven by insulin) and MAP (driven by norepinephrine), read the model's rollout response per stay-hour from dose_edit, and calibrate the kernel against my records-based impulse responses.

That only works if the model's state encodes recent doses linearly, including for doses beyond the training range. Next I'll pick 100 training stays with a wide spread of insulin and norepinephrine rates, including very high insulin and capped norepinephrine, for a states request. I'll then regress dose-history features on the states. A high R² that holds up at the largest observed doses would make the state a usable source for the correction. A poor or saturating fit would push me to a different mechanism.

```python
ins_all = np.nan_to_num(P['insulin'].values[:, col(0):col(24)+1]); ne_all = np.nan_to_num(P['norepinephrine'].values[:, col(0):col(24)+1])
rng = np.random.default_rng(0)
maxins = ins_all.max(1); maxne = ne_all.max(1); nins = (ins_all>0).sum(1); nne = (ne_all>0).sum(1)
art_all = np.isfinite(P['map_arterial'].values[:, col(0):col(24)+1]).mean(1) > 0.8
c1 = np.where((maxins>=5)&(nins>=15))[0]; c2 = np.where((nins>=15)&(maxins<5)&(nne==0))[0]
c3 = np.where((nne>=15)&art_all&(maxne>=1.0))[0]; c4 = np.where((nne>=15)&art_all&(maxne<0.6))[0]; c5 = np.where((nne>=10)&(nins>=10)&art_all)[0]
pick = list(rng.choice(c1,25,replace=False))+list(rng.choice(c2,15,replace=False))+list(rng.choice(c3,20,replace=False))+list(rng.choice(c4,15,replace=False))+list(rng.choice(c5,25,replace=False))
pick = sorted(set(pick)); print(len(pick))
sel_states = [int(ids[i]) for i in pick][:100]
json.dump(sel_states, open('/work/sel_states.json','w'))
print(json.dumps(sel_states))
```