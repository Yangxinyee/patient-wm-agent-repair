The ridge translation (R² 0.99) quantifies how badly v3 is aimed:

- **Insulin bundles:** predicted next finger glucose drops by about 1.0 mg/dL. That is roughly the intended direction, but small.
- **Non‑drug bundles:** the leak through u_const raises predicted finger glucose by about +4 mg/dL and lowers MAP by about −12 mmHg, on average.

The leak cancels between the two arms of a dose edit, which is why fwd_6 looked like "no effect". It would badly damage the factual fit, so v3 has to be replaced.

The ridge is only approximate for off‑manifold state changes. Still, the puzzle stands: an immediate shift of about −1 mg/dL per insulin bundle should accumulate over a rollout, given the model's ~0.96 belief persistence. Yet fwd_6 showed only −0.15 overall.

Before designing v4 I want two free checks:
1. The hour‑by‑hour profile of the fwd_6 rollout shifts, edited vs original. This shows whether an injected shift persists or dies out quickly in the rollout.
2. My stored truth impulse responses (hF, hA), to recall the temporal shape I must match.

If the edited rollout shift at hour 13 is noticeably negative and then fades, persistence in the rollout is short and the gain/shape must be designed around that. If it is flat near zero, the drug‑bundle injection is not reaching the rollout predictions at all.

```python
for m in ['original','edited']:
    de=pd.read_csv(f'/work/fwd_6/dose_edit_{m}.csv')
    if m=='original': print(de.columns.tolist()); print(de.head(3).to_string())
    cols=[c for c in de.columns if 'shift' in c or c in('insulin_x2','insulin_half','ne_x2','ne_half')]
    for c in cols[:6]:
        try: print(m,c,de.groupby('hour')[c].mean().round(2).to_dict())
        except Exception as e: print('err',c,e)
for nm in ['hF','hA','hL','hC']:
    if nm in globals(): print(nm,np.round(np.asarray(globals()[nm]),3))
```