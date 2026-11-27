The error is mostly a function of insulin rate, not bundle composition. The true projection jumps sharply between rate 1.5 (0.53) and rate 2.0 (2.75), and is smooth elsewhere. My rebuild puts that jump in a different place: errors of −0.6 to −0.8 at rates 0.5–1.5 and +0.3 to +0.4 at 2.0–2.5.

That points to bin assignment. A fine-bin edge for the insulin rate or dose probably sits at about 2, and my rebuild assigns the bins differently. It could be the edge side (left vs right), the edge table I used, or rate vs dose bins swapped.

So I'll print the vocab entries for the insulin and norepinephrine rate/dose edges and normalisation. I'll also print the bins my helper functions assign at rates 0.5–8.5, under both edge sides. If rate 2.0 falls exactly on an edge and my side convention is the other one, that is the bug. Otherwise I'll compare dose bins.

```python
print(list(VOCAB.keys()))
for k_ in VOCAB:
    v_=VOCAB[k_]
    if isinstance(v_,dict): print(k_, list(v_.keys())[:12])
cfgk=[k_ for k_ in VOCAB if 'edge' in k_.lower() or 'bin' in k_.lower() or 'norm' in k_.lower()]
for k_ in cfgk:
    v_=VOCAB[k_]
    if isinstance(v_,dict):
        for c_ in v_:
            if 'insulin' in str(c_) or 'norepi' in str(c_): print(k_, c_, json.dumps(v_[c_])[:600])
```