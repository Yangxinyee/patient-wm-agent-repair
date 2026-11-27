Insulin START atoms are verified too: unit id 3, rate_unit id 2, error 0.0003.

Norepinephrine END atoms match none of the four simple variants (errors 0.10–0.16). So END atoms carry something extra or different: a status/action/kq row, another event kind, or they may be encoded as observations (e_k) rather than actions. This matters for the correction:
- If END atoms carry the doubled rate, they also land in bin 16, and the correction needs a kind-specific version.
- If they carry no rate, the gate simply stays closed for them.

This run searches the END-atom structure more broadly:
- optional kq, status and action rows;
- every event kind;
- with or without rate/dose attributes;
- passing through encode.* as e_k instead of u_k.

A near-zero error identifies the structure.

```python
def avec3(rt, ds, attr, kq, st, ac, ek_):
    x = EM['code'][9] + EM['kind'][2] + EM['ekind'][ek_] + EM['bins.value'][0] + NB + EM['categorical.unit'][4] + EM['categorical.rate_unit'][3]
    x = x + (EM['bins.rate'][fb_('norepinephrine','rate',rt)] + EM['bins.dose'][fb_('norepinephrine','dose',ds)] + NW@np.array([0,0,zs_('norepinephrine','dose',ds),1,zs_('norepinephrine','rate',rt),1]) if attr else EM['bins.rate'][0] + EM['bins.dose'][0])
    if kq is not None: x = x + EM['kq'][kq]
    if st is not None: x = x + EM['categorical.status'][st]
    if ac is not None: x = x + EM['categorical.action'][ac]
    return x
itE2 = itE[:12]; resE = []
for attr in [True, False]:
    for kq in [None,0,1,2,3]:
        for st in [None,0,1]:
            for ac in [None,0,1]:
                for ek_ in [2,0,1,3,4]:
                    resE.append((zerr(itE2, lambda rt,ds: avec3(rt,ds,attr,kq,st,ac,ek_)), attr, kq, st, ac, ek_))
resE.sort(key=lambda t: t[0]); print('u-path best', [(round(t[0],4),)+t[1:] for t in resE[:4]])
def zerr_e(items, f): return float(np.mean([np.abs(zpo(c, uenc([f(rt,ds)], 'encode'), np.zeros(192))-XALL[jx,960:992]).mean() for (jx,c,rt,ds) in items]))
resEe = sorted((zerr_e(itE2, lambda rt,ds: avec3(rt,ds,attr,kq,None,None,2)), attr, kq) for attr in [True,False] for kq in [None,0,1,2,3])
print('e-path best', [(round(t[0],4),)+t[1:] for t in resEe[:3]])
```