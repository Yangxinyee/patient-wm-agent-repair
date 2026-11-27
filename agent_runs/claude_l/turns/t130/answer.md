The function sources are gone, since they were defined interactively. Their outputs and inputs should still be in kernel variables, though.

The e mismatch matters for any exact core step, not only at measurement bundles. Every step's cross-attention reads the memory of past e vectors.

From the run-105 printout, my probe appears to have mapped part of z onto fixed projections of e and the rest onto projections of u. That would let me test e-encoder hypotheses bundle by bundle against the tool's z values. I'll print the shapes and types of those leftover variables to reconstruct the setup.

```python
for k in ['P','Pc','Pe','Pu','Z7','Ev7','Vv7','single7','er_e','er_u','okn','okt','route','fix','res','res_','allres','recs','BUN','stays','rows','pz','zt','za','ee','ee_','eu','ea','ec','npz']:
    v=globals().get(k)
    try: sh=getattr(v,'shape',None) or (len(v) if hasattr(v,'__len__') else None)
    except Exception: sh=None
    print(k,type(v).__name__,sh, (str(v)[:80] if isinstance(v,(dict,str,int,float)) else ''))
```