Exact matches for both drugs' start atoms. Insulin uses unit row 3 and rate-unit row 2. Norepinephrine uses unit row 4 and rate-unit row 3. Both use drug-admin event kind and kq row 0, and carry dose and rate bins and z-scores.

Now the END atoms. I'll test the multi-atom action bundles (start + END, or END alone) under two hypotheses. In one, the END atom is the same as a start atom but with atom kind 2 and the infusion's rate and dose attached. In the other, it carries no attributes. An exact match under one of them completes the action embedding. It also settles whether doubled END atoms carry dose information, which matters because the dose edit doubles them too.

```python
CFG = {'insulin':(3,2), 'norepinephrine':(4,3)}
for ea in [True, False]:
    cfg = dict(CFG); cfg['end_attrs'] = ea
    for n_at in [1,2,3,4]:
        tl = [a for a in atests if len(a[3])==n_at]
        print('end_attrs', ea, 'n', n_at, a_err(tl[:20], cfg))
```