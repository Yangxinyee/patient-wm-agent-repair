Accounting for policy feedback gives a smooth, all-negative structural impulse response for a 1 u/h·1 h insulin dose: −1.95, −2.29, −1.94, −1.51, −1.15, −0.84, −0.55, −0.38 mg/dL at lags 1–8. It peaks at 2 h and decays by about 0.75 per hour. The feedback is small (an extra unit lowers later insulin by only 0.03–0.06), so the positive tail in the earlier IV run was a specification artefact. Under a sustained ×2 dose the cumulative response is about −10 to −11 mg/dL per u/h after 8 h. That implies a main readout of roughly −10 to −15 mg/dL, but it depends on the actual dose patterns.

Next I will record this and prepare the tuning stays. With these, a forward dose_edit request will show what the model currently predicts and the exact readout format.

```python
with open('/work/notes.md','a') as f:
    f.write("""- Reduced-form IRF of insulin deviation x_t on g_{t+s}, s=1..8: -1.97,-2.23,-1.74,-1.19,-0.79,-0.49,-0.24,-0.12.
  Policy feedback dr(m)=effect of x_t on r_{t+m}: 1.01, -.046,-.061,-.052,-.034,-.018,-.015,-.004.
  Recursive structural IRF h(k) (fixed doses), per 1 u/h for 1h: -1.95,-2.29,-1.94,-1.51,-1.15,-0.84,-0.55,-0.38 (k=1..8). Peak lag2, decay ~0.75/h.
  (2SLS distributed-lag gave positive lag7-8 -> likely misspecified; distrust.)
  Rough expectation: insulin x2 readout ~ -10 to -15 mg/dL (to be computed on actual dose patterns).
""")
tu = pd.read_csv('/data/tuning.csv')
print(tu.shape, tu.stay_id.nunique())
tu['h']=np.floor(tu.hour).astype(int)
ins_after = tu[(tu.event=='insulin')&(tu.hour>=12)].stay_id.unique()
ne_after = tu[(tu.event=='norepinephrine')&(tu.hour>=12)].stay_id.unique()
both = np.intersect1d(ins_after, ne_after)
print('tuning stays with insulin>=12:',len(ins_after),' norepi>=12:',len(ne_after),' both:',len(both))
rng=np.random.default_rng(0)
tstays = np.sort(tu.stay_id.unique())
sel1 = list(rng.choice(tstays, 100, replace=False))
json.dump([int(s) for s in sel1], open('/work/sel1.json','w'))
print(json.dumps([int(s) for s in sel1]))
```