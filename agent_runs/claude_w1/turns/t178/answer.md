The bytecode gives the exact register format. For each reading, gen_samples looks up the last reading of the same code and builds last = (1, age in minutes, z-scored last value, last value bin), or (0, 0, 0, 0) if there is none. It then calls wc_vec(state before the bundle, time gap to the previous bundle, code, last).

I now write my own generator, gen_rows_all, that does this for all four value codes, so I get head inputs for glucose_lab and map_cuff queries as well. It returns rows in the layout head_mean expects. Check: on two stays its finger and arterial head inputs must match gen_samples exactly. If not, I have the register or ordering wrong.

```python
VCODES = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
def gen_rows_all(df, St=None):
    B = bundles_of_stay(df, RU0)
    if St is None: St = run_stay(B)
    tk = np.array([b['t'] for b in B]); keyk = {int(round(t*60)): k for k, t in enumerate(tk)}
    meas = df[df.event.isin(VCODES)].sort_values('hour'); lastv = {}; out = []
    for hr, grp in meas.groupby('hour', sort=True):
        k = keyk.get(int(round(hr*60)))
        if k is None or k == 0:
            for r in grp.itertuples(): lastv[r.event] = (r.value, r.hour)
            continue
        for r in grp.itertuples():
            if r.event in lastv:
                lv_, lh_ = lastv[r.event]; zl, vb = zb(r.event, 'value', lv_); last = (1, (r.hour-lh_)*60, zl, vb)
            else: last = (0, 0, 0, 0); lv_ = np.nan
            out.append((r.event, r.stay_id, r.hour, lv_, r.value, 0, 0, 0, wc_vec(St[k-1], tk[k]-tk[k-1], r.event, last)))
        for r in grp.itertuples(): lastv[r.event] = (r.value, r.hour)
    return out
for s in H100[:2]:
    a = gen_rows_all(G3[s]); b = gen_samples(G3[s], pert=False)
    da = {(r[0], r[2]): r[8] for r in a}; diffs = [np.abs(da[(r[0], r[2])] - r[8]).max() for r in b if (r[0], r[2]) in da]
    print(s, 'all-rows', len(a), 'gs rows', len(b), 'matched', len(diffs), 'max diff %.2e' % (max(diffs) if diffs else -1), 'codes', dict(zip(*np.unique([r[0] for r in a], return_counts=True))))
```