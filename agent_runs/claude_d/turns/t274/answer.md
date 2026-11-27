There is no input history, so I cannot read back runs 236–238. What I have now: z = (x − median)/IQR clipped to ±10, and fine bin = 1 + the number of edges below x. So any doubled NE rate above 1.0 lands in bin 16, the top bin shared with insulin.

The key unknown for the encoder fix is the END atoms that drug records leave at end_hour. Whether they also go through `encode_action` carrying the doubled rate decides whether the gated correction must cover kind-2 atoms as well as starts. My `bundles_of` helper reproduced the state rows exactly, so its output for one stay should show what each bundle holds. I will print its structure around hour 12–14.

```python
B=bundles_of(TU[TU.stay_id==2020000000])
print(type(B), len(B))
b0=B[0] if isinstance(B,(list,tuple)) else B
print(type(b0)); 
try:
    print(B.head(12).to_string() if hasattr(B,'head') else [str(x)[:300] for x in B[:3]])
except Exception as ex: print(repr(ex)[:200])
```