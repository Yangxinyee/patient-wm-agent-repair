Doses are deterministic functions of rate: insulin dose equals rate, and norepinephrine dose is 4.8 × rate (80 kg for everyone). Each hour has at most one record per drug. So rate is the only dose variable that matters.

Next, the treatment pattern:
- whether infusions run in contiguous blocks once started;
- how the insulin rate depends on the glucose reading that hour (taken 5 minutes before the dose), the previous rate, and the last available glucose.

A dosing rule with substantial unexplained noise would give near-random variation for estimating effects. A rule driven by something unobserved, such as true glucose, means I'll need more care with confounding.

```python
P = Ptr.copy()
P = P.reset_index()
# contiguity of infusion blocks
def block_stats(col):
    g = P[P[col].notna()].groupby('stay_id').h
    first = g.min(); last = g.max(); n = g.size()
    contiguous = (last-first+1 == n)
    return first.describe(), contiguous.mean(), (last==24).mean()
for col in ['insulin','norepinephrine']:
    f, c, l = block_stats(col)
    print(col, "first hour:", f.to_string(), "\ncontiguous frac", c, "ends at 24 frac", l)
# hist of first hour
for col in ['insulin','norepinephrine']:
    g = P[P[col].notna()].groupby('stay_id').h.min()
    print(col, g.value_counts().sort_index().to_string())
```