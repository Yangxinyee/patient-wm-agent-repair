Two findings:
- **END atoms in e carry no rate, dose, bins or units**: only code, kind, event kind and `kq`. With that, the error is exactly zero. So the memory M holds no dose information at all; it only records that an infusion ended. A cross-attention dose memory is impossible, which rules out Plan A as conceived. The dose edit also leaves END atoms unchanged.
- **Measurements:** the unit swap is right, but a 0.23 median error remains, and neither tie rule nor bin offsets fix it.

To localise the remaining measurement error, I'll break it down by single-code bundles: cuff alone, arterial alone, finger alone. I'll also look at whether it depends on the value bin or the z-score.

```python
def Vfix(a):
    a=Vunit(a); a=Vend_bare_nou(a); return a
def per_bundle_err(var, idxs):
    out=[]
    for i in idxs:
        s,k=recs[i][0],recs[i][1]; atoms=BUN[s][k][1]
        ea=[var(dict(a)) for a in atoms if (not a['act']) or a['kind']==2]
        e=set_enc(W0, ea, 'encode').detach().numpy()
        out.append((i, tuple(sorted(a['code'] for a in ea)), np.abs(Pe@e-recs[i][4]).max(), ea))
    return out
pb=per_bundle_err(Vfix, idx_meas)
g=defaultdict(list)
for i,key,err,ea in pb: g[key].append(err)
for k_,v in sorted(g.items(), key=lambda kv:-len(kv[1]))[:10]: print(k_, len(v), 'median %.4f max %.4f'%(np.median(v),np.max(v)))
single=[(i,ea[0],err) for i,key,err,ea in pb if len(ea)==1]
for c in sorted(set(a['code'] for _,a,_ in single)):
    rows_=[(a['vb'],round(a['num'][0],3),round(err,4)) for _,a,err in single if a['code']==c]
    print('code',c, rows_[:8])
```