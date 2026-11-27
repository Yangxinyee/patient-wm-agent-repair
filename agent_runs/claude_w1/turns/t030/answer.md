Both rules are confirmed:
- **Rows:** a stay-hour (13–24) appears exactly when the record has a finger glucose reading (insulin edits) or an arterial MAP reading (norepinephrine edits).
- **Reached:** a row counts when its hour is later than the first dose at or after hour 12.

So the evaluation averages the model's shift over real measurement hours after the drug starts.

Next I'll compute my estimated true shift on exactly these rows. For each reached stay-hour t, ΔG_t = Σ_{s=12..t−1} h(t−s)·(f−1)·dose_s, using the deconvolved impulse responses and recorded doses, with f = 2 or 1/2. I'll also check whether the summary's mean_shift is pooled over stay-hours or averaged per stay, so my target is computed the same way.

```python
def true_shift(Pdf, stays, dcol, outcol, h, factor, rows):
    out=[]
    for sid in stays:
        s = Pdf[Pdf.stay_id==sid].set_index('h')
        d = s[dcol].reindex(range(-1,25)).fillna(0.0)
        for t in rows.get(sid, []):
            v = 0.0
            for sh in range(12, t):
                k = t - sh
                if 1 <= k <= len(h): v += h[k-1]*(factor-1)*d[sh]
            out.append((sid,t,v))
    return pd.DataFrame(out, columns=['stay_id','hour','true'])
res = {}
for edit, dcol, hh, f in [('insulin_x2','insulin',hi,2.0),('insulin_half','insulin',hi,0.5),('ne_x2','norepinephrine',hn,2.0),('ne_half','norepinephrine',hn,0.5)]:
    sub = de[(de.edit==edit)&de.reached]
    rows = sub.groupby('stay_id').hour.apply(list).to_dict()
    ts = true_shift(tuP, list(rows.keys()), dcol, None, hh, f, rows)
    m = sub.merge(ts, on=['stay_id','hour'])
    res[edit]=m
    print(edit, "model pooled", round(m['shift'].mean(),3), "per-stay avg", round(m.groupby('stay_id')['shift'].mean().mean(),3),
          "| true(est) pooled", round(m['true'].mean(),3), "per-stay", round(m.groupby('stay_id')['true'].mean().mean(),3))
```