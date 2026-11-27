The insulin dose noise looks purely random: log-scale SD 0.333, a within-stay lag-1 correlation of 0.002, and the variance of per-stay means matches what independent noise would give. So given the reading, the dose varies as if randomised. Regression adjustment should then identify the causal effect without needing the thresholds. The crash only happened because I rebuilt Q without the dG1 column.

Next I will rerun the regressions with dG1 restored. I am also adding outcomes 2 and 3 hours ahead (R_{t+2} − R_t, R_{t+3} − R_t) to see whether the effect builds up or fades. A per-u/h coefficient near −0.3 at one hour that grows at longer lags would mean insulin acts with delay or persistence.

```python
for k in [1,2,3]:
    Q[f'dG{k}'] = Q[f'glucose_finger_f{k}'] - Q.glucose_finger
D = Q[Q.hr>=0].copy()
gb = np.clip(D.glucose_finger.values, 60, 400)
Xs = [np.ones(len(D))] + [np.maximum(gb-k,0) for k in knots[:-1]] + [gb]
T = pd.get_dummies(D.tier).values[:,1:].astype(float)
for k in [1,2,3]:
    m = D[f'dG{k}'].notna().values
    for lab, X in [("smooth", np.column_stack(Xs + [D.ins.values])), ("tierFE", np.column_stack(Xs + [T, D.ins.values]))]:
        b,se = ols(D[f'dG{k}'].values[m], X[m], D.stay.values[m])
        print(f"dG{k} {lab}: dose coef {b[-1]:.3f} +- {se[-1]:.3f}   n={m.sum()}")
```