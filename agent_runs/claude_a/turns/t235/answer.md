The query and `with_code` input are built from `embed.code` and `embed.bins.*`, which the encoder also uses, so I must not edit them. `last_numeric` (192×8) and `last_status` are head-only. `type_emb` has 15 types, not 13 codes, so it cannot gate by code.

The design that follows: gate the readout with a linear discriminant on `with_code` and use two cancelling GELU units, so gate noise drops out for glucose_finger and the units stay off for other codes. The compensation for factual fit goes in head-only per-last-bin transition rows, which do not react to the dose edit.

Next I'll load my library files and run the teacher-forced loop on the 300-stay set. I also need to see what `prelude_parts` returns, since I must fit the write gate on prelude internals.
- **Ready to fit:** the post-attention residual or its norm is exposed per bundle and slot.

```python
G=globals()
for f in ['lib1.py','lib2.py','lib3.py']: exec(open('/work/'+f).read(), G)
print(type(dsE), [k for k in dsE.keys()][:20])
H,Out,st = tf_loop(PO, dsE, keep=('hf','E'))
print({k:(tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k,v in st.items()})
r = prelude_parts(PO, st['hf'][:6], dsE['e'][:6], dsE['u'][:6], st['z'][:6])
items = r.items() if isinstance(r,dict) else enumerate(r)
print(type(r).__name__, [(k, tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k,v in items][:12])
ri=bundle_rates(dsE,'insulin'); rn=bundle_rates(dsE,'norepinephrine')
print(len(ri), (ri>0).sum(), (rn>0).sum(), ((ri>0)&(rn>0)).sum())
```