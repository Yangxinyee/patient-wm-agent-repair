The finger-glucose cost is concentrated, not spread evenly. Over the three batches (4205 rows, 44.7 nats):

| Predicted-mean shift | Rows | Cost per row | Total (nats) |
|---|---|---|---|
| −3 to −5 | 279 | +0.060 | 16.6 (37% of all) |
| −5 to −8 | 100 | ≈ 0 | ≈ 0 |
| +5 to +8 | 62 | +0.12 | 7.4 |
| above +8 | 22 | +0.12 | 2.6 |

The pattern is not monotonic in the shift. That points to particular situations where the readout misreads the dose, rather than general noise, and an identifiable failure mode might be gated out cheaply.

I'll describe the rows in the costly bins using the tuning grid:
- their previous finger value and tier;
- the insulin dose given the hour before;
- the true deviation;
- the hour.

Then I'll compare them with the cheap −5 to −8 bin and the near-zero rows.

```python
gfT = Gtu['gf']; inT = Gtu['ins']
s_ = FR[FR.event=='glucose_finger'].copy()
hh_ = np.floor(s_.hour.values).astype(int) - 1
s_['gp'] = [gget2(gfT, s, h) for s, h in zip(s_.stay_id.values, hh_)]
s_['dose'] = np.nan_to_num([gget2(inT, s, h) for s, h in zip(s_.stay_id.values, hh_)])
s_['tier'] = tier5(s_.gp.values)
s_['dev'] = s_.dose - EtabL[s_.tier.values]
s_['res'] = s_.observed - (pd.concat([pd.read_csv(f'/work/fwd_{k}/factual_original.csv') for k in [13,14,15]], ignore_index=True).loc[s_.index, 'predicted_mean'].values)
s_['bin'] = pd.cut(s_.dpm, [-100, -5, -3, -1.5, -0.3, 0.3, 1.5, 3, 5, 100])
agg_ = s_.groupby('bin', observed=True).agg(n=('dn','size'), dn=('dn','mean'), gp=('gp','mean'), dose=('dose','mean'), dev=('dev','mean'),
        res=('res','mean'), hour=('hour','mean'), frac_t3=('tier', lambda t: (t==3).mean()), frac_t0=('tier', lambda t: (t==0).mean()), frac_nodose=('dose', lambda d: (d==0).mean()))
print(agg_.round(2).to_string())
```