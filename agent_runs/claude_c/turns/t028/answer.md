The tool's mean_shift equals the pooled mean over reached stay-hours (model −1.85), so that is the aggregation. On the same 100 stays my estimates imply:

| Edit | Estimated truth | Model now |
|---|---|---|
| Insulin ×2 | −13.1 mg/dL | −1.85 |
| Insulin ×½ | +6.5 | +0.78 |
| NE ×2 | +4.75 mmHg | −0.50 (wrong sign) |
| NE ×½ | −2.4 | −0.60 |

The model's insulin response is about 1/7 of the truth. For NE it is roughly zero or of the wrong sign.

Next I will compare the shape of the response over time, model versus truth, by hours since the first dose. A constant fraction would suggest a uniformly weak dose pathway. A response that starts right and then fails to accumulate, or that saturates, would point to off-distribution behaviour or a diagnostic offset. I will also compute the estimated true readouts over all 400 tuning stays and 3,000 training stays, to see how much the target varies by sample.

```python
tt['first'] = np.nan
def first_dose_hour(Pp, sid, drug):
    d = Pp.loc[sid][drug]; d = d[(d.index>=12)&d.notna()]
    return d.index.min() if len(d) else np.nan
fd = {(sid,dr): first_dose_hour(Ptu, sid, dr) for sid in tt.stay_id.unique() for dr in ['insulin','norepinephrine']}
tt['since'] = [h - fd[(s,'insulin' if e.startswith('insulin') else 'norepinephrine')] for s,h,e in zip(tt.stay_id, tt.hour, tt.edit)]
print(tt.groupby(['edit','since'])[['true','model']].mean().round(2).unstack(0).to_string())
# population-level true readouts (requires 'reached' rows defined by recorded readings) 
def pop_true(Pp, stays, b_ins, b_ne):
    out = {k:[] for k in ['insulin_x2','insulin_half','ne_x2','ne_half']}
    for sid in stays:
        S = Pp.loc[sid]
        for drug, code, b, keys in [('insulin','glucose_finger',b_ins,('insulin_x2','insulin_half')), ('norepinephrine','map_arterial',b_ne,('ne_x2','ne_half'))]:
            d = S[drug].values; hrs = S.index.values; dd = dict(zip(hrs, d))
            dos = [s for s in range(12,25) if not np.isnan(dd.get(s,np.nan))]
            if not dos: continue
            f0 = min(dos)
            for t in range(13,25):
                if t<=f0 or np.isnan(S[code].get(t,np.nan)): continue
                e = sum(dd[s]*b[t-s-1] for s in dos if s<t and t-s<=len(b))
                out[keys[0]].append(e); out[keys[1]].append(-0.5*e)
    return {k:(np.mean(v), len(v)) for k,v in out.items()}
print("tuning all 400:", pop_true(Ptu, tu_ids, b_ins, b_ne))
rng = np.random.default_rng(0)
trs = rng.choice(Ptr.index.get_level_values(0).unique(), 3000, replace=False)
print("train 3000:", pop_true(Ptr, trs, b_ins, b_ne))
```