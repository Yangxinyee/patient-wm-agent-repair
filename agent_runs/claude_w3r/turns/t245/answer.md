`u_of` does **not** reproduce every dose bundle: the maximum error over all 2406 is 4.8, while row 0 matched. So some synthetic ×2 and ×½ tokens were wrong, and the augmented fits are contaminated:
- the presence gate got worse (resid 0.28);
- the ×2 slopes came out at 0.69–0.85.

Most likely `u_of` mishandles one-drug bundles, or real bundles carry atoms `u_of` doesn't model.

**Conclusion:** I need correct edited tokens before fitting anything else. The right source is the real bundles: `edit_atoms(atoms, code, mult)` followed by `enc_eu(atoms)`. Run 170 probably did this already: UX, 9918 rows, with YXi and YXn.

**This run:**
1. Locate the `u_of` errors by drug combination.
2. Inspect UX, YXi, YXn and any 9918-long label array.
3. Test `edit_atoms` and `enc_eu` on one real dose bundle.

**Test:** if UX already holds factual plus edited dose vectors with labels, I refit the reads on those directly. If not, I generate them from real bundles.

```python
errs=np.array([np.abs(n1u(pa[i,0]*ins_p[i],pa[i,1]*ne_p[i])-U_act[i]).max() for i in range(len(pa))])
for a in [0,1]:
    for b in [0,1]:
        m=(ins_p==bool(a))&(ne_p==bool(b))
        if m.sum(): print('ins',a,'ne',b,'n',int(m.sum()),'err pct50/90/max',np.round(np.percentile(errs[m],[50,90,100]),4))
L=[(k,getattr(v,'shape',None)) for k,v in globals().items() if hasattr(v,'shape') and len(getattr(v,'shape',()))>0 and v.shape[0]==9918]
print('9918-long arrays:', L)
for nm in ['YXi','YXn']:
    v=globals().get(nm); 
    if v is not None: print(nm, np.round(np.asarray(v)[:8],3).tolist(), 'mean %.3f'%np.mean(v))
try: brief(edit_atoms,4); brief(enc_eu,4)
except Exception as e: print(e)
B_,rows_,EU_=stay_data(int(stays_u[0])) if 'stays_u' in globals() else (None,None,None)
ka=[k for k,b in enumerate(B_) if any(str(a[0]) in ('insulin','norepinephrine') for a in b[1])][:1]
print('example bundle', ka, B_[ka[0]] if ka else None)
```