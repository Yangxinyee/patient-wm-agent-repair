The negative control finds no hidden-information dosing. The insulin policy explains 82% of the rate variance from recorded history (residual SD 0.49 U/h). The residual does not predict the same-hour lab glucose once the finger reading is held fixed:
- all rows: 0.03 ± 0.12 mg/dL per U/h;
- glucose > 160: 0.25 ± 0.16 mg/dL per U/h.

Even taking the upper value, that is about 0.1 mg/dL per residual SD, which is negligible. So insulin appears to be dosed on recorded readings plus independent noise, and that noise is a usable natural experiment for the causal effect.

There is an implication for the edit. In teacher-forced prediction the dose adds little beyond the visible readings except its causal effect. In the rollout the readings are hidden, so a recorded or edited dose looks like evidence about glucose. Removing that evidential response should therefore cost little factual NLL.

Next I'll estimate the causal effect. I'll regress the next-hour finger glucose change on current and previous glucose, the predicted dose and the policy residual. If the residual coefficient is clearly negative, insulin lowers glucose; near zero means no effect. I'll also check whether the residuals persist within a stay. A stay-level dosing style would be a second source of variation and could hide its own confounding.

```python
m = m.sort_values(['stay_id','ih'])
m['g_next'] = m.groupby('stay_id').glucose_finger.shift(-1)
m['ih_next'] = m.groupby('stay_id').ih.shift(-1)
q = m[(m.ih_next==m.ih+1) & m.g_prev.notna()].copy()
q['dg'] = q.g_next - q.glucose_finger
for lab, sub in [('all', q), ('ins>0', q[q.insulin>0]), ('g>160', q[q.glucose_finger>160])]:
    beta, se = ols(sub.dg.values, [sub.glucose_finger, sub.g_prev, sub.ins_hat, sub.ires])
    print(lab, len(sub), "coef [c, g, gprev, ins_hat, ires]:", beta.round(3), se.round(3))
# residual autocorrelation within stays
q['ires_lag'] = q.groupby('stay_id').ires.shift(1)
z = q.dropna(subset=['ires_lag']); z = z[z.insulin>0]
print("resid autocorr:", np.corrcoef(z.ires, z.ires_lag)[0,1])
sm_ = q[q.insulin>0].groupby('stay_id').ires.agg(['mean','count']); sm_=sm_[sm_['count']>=5]
print("stay-mean resid sd:", sm_['mean'].std(), "expected if iid:", (q[q.insulin>0].ires.std()/np.sqrt(sm_['count'])).mean())
```