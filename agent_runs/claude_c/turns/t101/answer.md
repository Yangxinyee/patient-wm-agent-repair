Run 89 settles where the loss happens. Under the x5 numeric-input edit, the dose slope in z26 rose only from 1.04 to 1.25. The dose slopes of the state changes (`out` and h) did not move at all. So the scaled z-score inputs barely reach u_k: the action encoder carries the dose mostly through the fine-bin embeddings.

Next I'll look at the rate and dose bins. For each drug I'll check which bins factual, x2 and x1/2 rates fall into. The embed.bins.rate/dose rows 1-16 only enter drug atoms, so row edits there are drug-bundle-selective. The check is whether the bin index tracks log-rate regularly enough for a "linear trend in bin index" edit to make sense.

```python
for c in VOCAB['numeric']:
    nm = c.get('code', c.get('name', '?'))
    if nm in ('insulin','norepinephrine'):
        print(nm, {k: (np.round(v,3).tolist() if isinstance(v,list) else v) for k,v in c.items() if k!='code'})
```