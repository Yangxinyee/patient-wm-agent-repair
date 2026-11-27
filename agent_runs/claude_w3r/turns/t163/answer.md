The readout pools over "reached" stay-hours rather than averaging per stay first: 484 stay-hours over 55 stays for insulin and 238 over 21 stays for norepinephrine on sel1. Next I'll verify that pooled mean against the summary. Then I'll compute records-based targets for all four edits on exactly those rows, from the tuning records and my structural IRFs (insulin lags 1–12 summing to about −12.9; norepinephrine about +9.9). These targets are what I'll calibrate the channels against.

```python
TU = pd.read_csv('/data/tuning.csv'); sel1 = json.load(open('/work/sel1.json'))
for ed in D1.edit.unique():
    q = D1[(D1.edit==ed)&(D1.reached)]; print(ed, 'pooled mean', round(q['shift'].mean(),3), 'n', len(q))
IRF_I = np.array([-2.0,-2.45,-2.1,-1.7,-1.3,-1.0,-0.7,-0.5,-0.35,-0.25,-0.25,-0.3])
IRF_N = np.array([3.45,2.79,1.79,1.09,0.50,0.22,0.26,-0.14,0.07,-0.11,0.20,-0.18])
def targets(D_, recs_):
    dr_ = recs_[recs_.event.isin(['insulin','norepinephrine'])].copy(); dr_['h'] = np.floor(dr_.hour).astype(int)
    RR = {ev: dr_[dr_.event==ev].groupby(['stay_id','h']).rate.sum() for ev in ['insulin','norepinephrine']}
    out_ = {}
    for ed, ev, irf, mult in [('insulin_x2','insulin',IRF_I,2),('insulin_half','insulin',IRF_I,.5),('ne_x2','norepinephrine',IRF_N,2),('ne_half','norepinephrine',IRF_N,.5)]:
        q = D_[(D_.edit==ed)&(D_.reached)]; vals = []
        for s, t in zip(q.stay_id, q.hour):
            vals.append(sum(irf[k-1]*(mult-1)*float(RR[ev].get((s, t-k), 0.0)) for k in range(1,13) if t-k >= 12))
        out_[ed] = (round(float(np.mean(vals)),3), len(vals))
    return out_
print(targets(D1, TU))
```