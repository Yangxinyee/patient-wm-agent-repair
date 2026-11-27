The two token directions are unusable. The least-squares direction for glucose has norm about 5×10⁵, and adding it to the z token at insulin bundles changed the next prediction by exactly 0. The NE direction (norm 105) gives only about a seventh of its linear prediction when doubled. So the z token is a weak, saturating channel into the prelude. My saved edit has absurd z_in columns and must be replaced before any forward pass.

The gates and decoders are good and remain usable:
- the insulin deviation decoder has CV R² 0.93 and error sd 0.34 at recorded doses;
- the NE rate decoder has R² 0.98;
- all other z dims are unchanged.

To choose an injection point I need the exact prelude computation: whether tokens are normalised, how slots read tokens, and how the drug token u enters. I also want to know what the value head's glucose-specific input (last_numeric column 0) is, since a direct head input could be a much stronger lever. I'll print the relevant lines of ARCHITECTURE.md compactly.

```python
A_=open('/model/ARCHITECTURE.md').read().split('\n')
keys=('prelude','token','z_in','last_numeric','numeric_in','b_in','delta','encode_action','slot','attend','LayerNorm','LN(')
hit=[i for i,l in enumerate(A_) if any(k in l for k in keys)]
txt=[]; seen=set()
for i in hit:
    for j in range(max(0,i-1),min(len(A_),i+2)):
        if j not in seen: seen.add(j); txt.append(A_[j].strip()[:160])
print(len(A_), len(hit)); print('\n'.join(t for t in txt if t)[:4000])
```