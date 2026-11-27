The vocab gives the z-scores and the bin edges.

Insulin ×2 can push z_dose up to about 8.5, still inside the clip at 10, and bin 16. Norepinephrine ×2 creates doses above 4.8, which never occur in training because the rate is capped at 1.0. So the norepinephrine edit is out of distribution.

To re-implement the embedding I need to know which categorical attributes each atom carries (unit, route, action, status) and how my bundles represent drug and END atoms. Next I'll print the start of the data README and one drug bundle from my rebuilt bundles.

```python
rd=open('/data/README.md').read(); print(len(rd)); print(rd[:2500])
bl=bundles_B[list(bundles_B.keys())[0]]
for t_,ats in bl:
    if any(a_['code'] in ('insulin','norepinephrine') for a_ in ats):
        print(t_, ats); break
```