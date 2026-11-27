The sliding-scale expectation per glucose_finger band is 0.013 (≤150), 1.03, 2.03, 3.03, 4.10 and 6.21 units/h. Insulin is given in about 97% of readings above 150. So the expected dose is essentially the base rate of the band, scaled by the multiplier mean and the probability of being given.

I will now compute, for every glucose reading in the training and tuning tables:
- R (true kernel times dose history);
- C (kernel times expected dose from earlier glucose_finger readings);
- for the MAP readings, the norepinephrine analogue R_N.

I will then measure the cross-validated R² of ridge readouts from q for R, C and T = R − C on the training stays.

```python
def hourly_maps(df):
    I = {}; N = {}; G = {}
    for r_ in df[df.event == 'insulin'][['stay_id', 'hour', 'rate']].itertuples():
        I.setdefault(r_.stay_id, {})[int(np.floor(r_.hour))] = r_.rate
    for r_ in df[df.event == 'norepinephrine'][['stay_id', 'hour', 'rate']].itertuples():
        N.setdefault(r_.stay_id, {})[int(np.floor(r_.hour))] = r_.rate
    gg = df[df.event == 'glucose_finger'][['stay_id', 'hour', 'value']]
    for r_ in gg.itertuples():
        G.setdefault(r_.stay_id, {})[int(round(r_.hour))] = r_.value
    return I, N, G
def add_targets(RTz, I, N, G, K=12):
    RI, CI, RN = [], [], []
    for s, m in zip(RTz.stay_id, RTz.minute):
        t = int(round(m / 60.0)); Is = I.get(s, {}); Ns = N.get(s, {}); Gs = G.get(s, {})
        RI.append(sum(KI[k - 1] * Is.get(t - k, 0.0) for k in range(1, K + 1)))
        CI.append(sum(KI[k - 1] * cexp(Gs[t - k]) for k in range(1, K + 1) if (t - k) in Gs))
        RN.append(sum(KN[k - 1] * Ns.get(t - k, 0.0) for k in range(1, K + 1)))
    RTz = RTz.copy(); RTz['RI'] = RI; RTz['CI'] = CI; RTz['TI'] = np.array(RI) - np.array(CI); RTz['RN'] = RN
    return RTz
Itr, Ntr, Gtr = hourly_maps(trg)
RT4 = add_targets(RT4, Itr, Ntr, Gtr)
from sklearn.model_selection import cross_val_predict
gl_ = RT4.event.isin(['glucose_finger', 'glucose_lab']).values
Xq = q4.numpy()[gl_]
for tcol in ['RI', 'CI', 'TI']:
    y_ = RT4[tcol].values[gl_]
    for al in [1.0, 10.0]:
        pr_ = cross_val_predict(Ridge(alpha=al), Xq, y_, groups=RT4.stay_id.values[gl_], cv=GroupKFold(5))
        print(tcol, 'alpha', al, 'sd %.2f' % y_.std(), 'CV R2 %.3f' % (1 - ((y_ - pr_) ** 2).mean() / y_.var()))
```