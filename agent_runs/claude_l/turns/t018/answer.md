A one-hour insulin pulse has a transient effect.
- **Shape:** removing the clinicians' feedback, which is small (D_k ≈ −0.05), gives an open-loop response of −2.06, −2.45, −2.18, −1.77, −1.29, −1.02, −0.76, −0.61, −0.42, −0.24, −0.22, −0.12 mg/dL at 1–12 h per 1 U/h for one hour. It peaks at 2 h, decays with a timescale of about 4 h, and integrates to about −13 mg/dL·h. So glucose mean-reverts toward a baseline that insulin does not shift, while the baseline itself wanders like a random walk.
- **Implied effect of insulin x2:** for a stay steadily on 2 U/h, glucose would be 4.1 mg/dL lower at hour 13. The gap would approach 26 mg/dL by hour 24, with a 13–24 average of about −20.

This holds only if the effect is linear in dose and homogeneous. Because x0, x1.5 and x3 are also scored, I need to check the dose-response shape and whether the effect depends on glucose level. I'll compute the first three hours of the local-projection effect in subgroups by predicted rate level and by glucose level. A per-unit effect that is similar across rate levels would support linearity in dose. One that scales with glucose would mean the effect is multiplicative in G.

```python
base_df = d.loc[base].copy()
for k in range(1,4):
    base_df[f'gk{k}'] = gsh[k].loc[base].values
base_df['y3'] = base_df[['gk1','gk2','gk3']].mean(axis=1, skipna=False) - base_df.glucose_finger
bd = base_df[base_df.y3.notna()]
def eff(sub):
    b_, s_ = ols(sub.y3.values, [sub.glucose_finger, sub.g_prev, sub.ins_hat, sub.ires]); return b_[-1], s_[-1], len(sub), sub.ires.std()
print("overall", eff(bd))
bd['rbin'] = pd.cut(bd.ins_hat, [-1,0.5,1.25,2,3,4.5,20])
for k_, sub in bd.groupby('rbin'): print('rate', k_, np.round(eff(sub),3))
bd['gb'] = pd.cut(bd.glucose_finger, [0,150,170,190,210,240,400])
for k_, sub in bd.groupby('gb'): print('gluc', k_, np.round(eff(sub),3))
# residual sd vs rate level
print(bd.groupby('rbin').ires.std())
```