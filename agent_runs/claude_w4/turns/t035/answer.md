## Step 33: the readout rows, and the true targets for these stays

**What the last result showed.**
- The per-row file has one row per stay × hour × edit, but only at hours 13–24 where the record has a finger-glucose reading (for insulin edits) or an arterial-MAP reading (for norepinephrine edits).
- A `reached` flag marks hours after the drug had run from hour 12 on.
- Each row holds the predicted shift and the factual predicted mean.

**What I will do next.**
1. Check how the summary aggregates: a pooled mean over stay-hours, or a mean of per-stay means. Match it against −0.348.
2. Check that "reached" means "some dose at or after hour 12 started before this hour".
3. Compute my predicted true shift for every row: the sum over lags of h(lag) × Δdose, using the recorded doses and the finger kernel (insulin) or the arterial kernel (norepinephrine).
4. Aggregate the same way as the tool, to get target values for all four edits on these stays.

**What would count as evidence.**
- If the pooled mean reproduces −0.348, the readout is a mean over stay-hours.
- My targets will show how far off the model is, in absolute terms.

```python
fw['reached'] = fw.reached.astype(bool)
for e in fw.edit.unique():
    q = fw[(fw.edit==e)&fw.reached]
    print(e, "pooled mean:", q['shift'].mean().round(3), " mean of stay means:", q.groupby('stay_id')['shift'].mean().mean().round(3), " n rows", len(q))
# check reached definition
PTr = PT.reset_index()
def first_dose_from12(sid, col):
    d = PTr[(PTr.stay_id==sid)&(PTr.h>=12)&(PTr[col]>0)]
    return d.h.min() if len(d) else np.nan
chk = fw[fw.edit=='insulin_x2'].copy()
chk['fd'] = chk.stay_id.map({s:first_dose_from12(s,'ins_rate') for s in chk.stay_id.unique()})
print(pd.crosstab(chk.reached, chk.hour > chk.fd))
chk2 = fw[fw.edit=='ne_x2'].copy()
chk2['fd'] = chk2.stay_id.map({s:first_dose_from12(s,'ne_rate') for s in chk2.stay_id.unique()})
print(pd.crosstab(chk2.reached, chk2.hour > chk2.fd))
# true predicted shifts using kernels
hF = np.array(bF)   # insulin finger kernel lags 1..12
bA, seA, _ = irf_ne(xn2[xn2.art_stay], 'map_arterial', 12)
hA = np.array(bA)
def true_shift(rows, col, h, mult):
    out = []
    for _, r in rows.iterrows():
        d = PTr[(PTr.stay_id==r.stay_id)]
        doses = d.set_index('h')[col]
        s = r.hour; tot = 0.0
        for t in range(12, s):
            lag = s - t
            if 1 <= lag <= len(h): tot += h[lag-1] * (mult-1) * doses.get(t, 0.0)
        out.append(tot)
    return np.array(out)
for e, col, h, mult in [('insulin_x2','ins_rate',hF,2.0),('insulin_half','ins_rate',hF,0.5),('ne_x2','ne_rate',hA,2.0),('ne_half','ne_rate',hA,0.5)]:
    q = fw[(fw.edit==e)&fw.reached].copy()
    q['true'] = true_shift(q, col, h, mult)
    print(e, "model:", q['shift'].mean().round(3), " my-true:", q.true.mean().round(3))
```