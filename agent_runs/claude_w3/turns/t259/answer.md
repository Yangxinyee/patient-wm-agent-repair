The infusion encoder is two 192→768→192 MLPs: a per-atom `mlp1`, then a sum divided by √n plus a size term, then `mlp2` and a LayerNorm. The atom's rate enters only through `embed.numeric` (6 numeric inputs) and the rate-bin table.

Next I'll read the vocabulary's numeric section, the score used for code 8 and code 9 rates and its note. That tells me whether a doubled rate shifts the score in proportion or by a constant. A linear score (rate minus centre, over scale) would allow a proportional channel. A log score would not.

```python
print(str(VOCAB['numeric_note'])[:400])
nu=VOCAB['numeric']; print(type(nu).__name__, list(nu.keys())[:12] if isinstance(nu,dict) else len(nu))
for k in list(nu.keys())[:12] if isinstance(nu,dict) else []:
    print(k, str(nu[k])[:160])
print(str(VOCAB['codes'])[:300])
```