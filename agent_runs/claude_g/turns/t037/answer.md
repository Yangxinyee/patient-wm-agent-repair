As expected, the edited model gives exactly 0.000 for all four readouts (insulin x2 and x1/2 over 815 stay-hours; norepinephrine x2 and x1/2 over 786). Dose size no longer reaches the model by any path, and its factual NLL is unchanged (+0.00001). I am logging this in the notebook.

The remaining risk is in my estimate of the truth, not the edit. For insulin I have tested each lag separately. The readout, though, asks about a cumulative quantity: all extra insulin from hour 12 up to each hour 13–24. So I will run a direct cumulative randomisation test on the training stays:
- regress G(t) − G(12) on X(t), the summed random parts of the insulin doses from hour 12 to t−1, controlling for G(12), the predictable dose part and hour;
- convert the slope into an implied insulin x2 readout with a confidence interval, using the recorded cumulative doses.

An implied readout statistically indistinguishable from 0, with a CI of a few tenths of a mg/dL, would make exact 0 the right target. A significantly non-zero slope would mean the cumulative effect differs from the sum of single-lag effects, and I would revisit.

```python
open('/work/notes.md','a').write("""
- fwd_2 factual both rand100 tuning: NLL orig 1.49270 vs edited(A) 1.49271 (+0.00001). fwd_3 dose_edit edited(A): all four readouts exactly 0.000.
- NE DML lags: -0.35(1),-0.38(2),-0.27(3),-0.13(4),-0.12(6),+0.06(8),+0.25(10),+0.58(12) per mcg/kg/min; NEGATIVE CONTROL (cuff MAP at h, pre-dose)
  gives -0.54+-0.21 -> titration tracks latent MAP -> short-lag negative is confounding. NE effect ~0 (implied |readout| < 0.5 mmHg).
- DML insulin k=1: 0.006+-0.038. BELIEF: both drugs inert; target readouts all 0; Edit A achieves exactly 0 with intact NLL.
""")
# cumulative randomization test for insulin
Pi = Q[['glucose_finger','glucose_lab','ins']].copy()
Pi['hh'] = Pi.index.get_level_values('h')
gl = Pi['glucose_finger'].fillna(Pi['glucose_lab'])
Pi['g_used'] = gl
Pi['base'] = base_ins(np.nan_to_num(Pi['g_used'].values, nan=0))
mu_m = d.groupby('base').m.mean().to_dict(); print('E[m|base]', {k: round(v,3) for k,v in mu_m.items()})
Pi['pred'] = Pi['base'].map(lambda b: b*mu_m.get(b,1.05) if b>0 else 0.0)
Pi['e'] = np.where(Pi['ins']>0, Pi['ins']-Pi['pred'], 0.0)
Pi.loc[Pi['g_used'].isin([150,180,220,260,300]), 'e'] = 0.0  # ambiguous thresholds: drop
Pi['e_late'] = np.where(Pi['hh']>=12, Pi['e'], 0.0); Pi['I_late'] = np.where(Pi['hh']>=12, Pi['ins'], 0.0); Pi['p_late']=np.where(Pi['hh']>=12, Pi['pred'],0.0)
gp = Pi.groupby(level='stay_id')
Pi['X'] = gp['e_late'].cumsum() - Pi['e_late']   # sum over s=12..t-1
Pi['D'] = gp['I_late'].cumsum() - Pi['I_late']
Pi['Pp'] = gp['p_late'].cumsum() - Pi['p_late']
g12 = Pi.xs(12, level='h')['glucose_finger']
Pi = Pi.join(g12.rename('g12'), on='stay_id')
R = Pi[(Pi.hh>=13)&(Pi.hh<=24)&Pi['glucose_finger'].notna()&Pi['g12'].notna()&(Pi['D']>0)].copy()
z12 = (R.g12.values-200)/50
Xr = np.column_stack([np.ones(len(R)), z12, np.maximum(z12+1,0), np.maximum(z12,0), np.maximum(z12-1,0), R.hh.values, R.Pp.values, R.X.values])
y = (R.glucose_finger - R.g12).values
b, se, _ = ols(y, Xr, R.index.get_level_values('stay_id').values)
print('cumulative slope per unit of cumulative random insulin:', round(b[-1],4), '+-', round(se[-1],4), ' n=', len(R))
print('mean D over these stay-hours:', R.D.mean().round(2), ' -> implied x2 readout:', round(b[-1]*R.D.mean(),3), '+-', round(se[-1]*R.D.mean(),3))
```