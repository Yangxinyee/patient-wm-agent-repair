The comparison crashed because one of the candidate dicts holds a tensor of a different shape under a standard name. It is probably an edited or scaffold variant, not a weight copy. I'll guard the comparison by shape and rerun the same three checks.

```python
ok={}
for nm in ['P','PO','PS','P_','Po','W','Wt','D_','TJ']:
    v=globals().get(nm)
    if isinstance(v,dict) and 'core.c_out.weight' in v:
        d=0.0; bad=0
        for n in NAMES:
            a=v[n]; a=a if torch.is_tensor(a) else torch.tensor(np.asarray(a))
            b=load_original(n)
            if tuple(a.shape)!=tuple(b.shape): bad+=1; continue
            d=max(d,float((a.float()-b).abs().max()))
        ok[nm]=(round(d,5),bad)
print('max diff vs original (diff, n_shape_mismatch):',ok)
pat={}
for n in NAMES:
    k=re.sub(r'\.\d+\.','.#.',n); pat.setdefault(k,[]).append(tuple(SHAPES[n]))
print(len(pat),'patterns')
print(' ; '.join(f"{k}{v[0]}x{len(v)}" if len(v)>1 else f"{k}{v[0]}" for k,v in pat.items()))
```