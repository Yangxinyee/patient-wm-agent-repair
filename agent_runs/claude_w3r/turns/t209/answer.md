Run 196 shows how an injected state component decays, with clear consequences for the lag design.

| Injection direction | Left after one bundle (55 min) | Left after two bundles |
|---|---|---|
| Generic or low-variance y direction | 25–37% | 7–17% |
| The 10 fastest-decaying dimensions | 0.5% | 0.4% |

The fast-dimension injection still moves `out` fully at the injection step (|Δout| 0.13).

So:
- **Lag 1:** a fast-dimension injection at the action bundle is visible to the next reading's head and then vanishes. That makes a clean, non-persistent lag-1 channel.
- **Lag 2:** persistence can only supply about 0.1–0.17 of b1, far short of the ≈0.47 needed. So I'll supply lag 2 from the END atom through head1, with weights c1 = 3.45 and c2 = 0.72 for norepinephrine.

Before building, I need the right compensation forms (insulin band term, norepinephrine intercept). These come from the factual residuals in fwd_3, so I'll check which DataFrames hold them.

```python
for nm_ in ['F3','fac','MT','D1','DH','fsa','fo','TG','TM','TU','M_','ev0','ev2','rec_gf','dr','q','rec']:
    v=globals().get(nm_)
    if isinstance(v,pd.DataFrame): print(nm_, v.shape, list(v.columns)[:30])
print(sorted(os.listdir('/work/fwd_3')), open('/work/fwd_3/summary.json').read()[:600])
```