`PS` is only a kill scaffold: every changed entry is zero, and the memory dims keep their original a_in/flow.a. So the write/read wiring from runs 143–153 is gone and has to be rebuilt. This time the source goes in a file.

The original memory dims (155, 171) already have almost no flow decay (flow.a ≈ −16, i.e. about 4e-4 per hour). They are rewritten at each bundle only through abar ≈ 0.34, so setting a_in very negative will make them persistent.

To rebuild the wiring correctly I need the exact computation order. That covers the prelude token layout and which norms feed which projections, and in the head the order of out_norm, query_norm and the value MLP. So next I'll re-read /model/ARCHITECTURE.md, first part only.

```python
s=open('/model/ARCHITECTURE.md').read()
print(len(s))
print(s[:5500])
```