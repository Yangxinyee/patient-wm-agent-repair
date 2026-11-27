With the correct XALL rows, one combination reproduces the stored z posterior almost exactly: mean error 0.0005 against values of about 0.25. That combination is:
- unit id 4 and rate_unit id 3;
- e_k equal to the empty-set encoding for drug-only bundles.

My atom embedding, set encoder, ctx and z_post are therefore verified for norepinephrine START atoms. The norepinephrine atom vector is code + kind + ekind + rate/dose/value bins + numeric (z_dose, 1, z_rate, 1) + bias + unit[4] + rate_unit[3].

To design the bin-16 correction, I still need to know:
1. Whether END atoms carry rate and dose. Doubling would then hit them as well, and the correction would need a kind-specific version.
2. Insulin's unit ids, to check that the gate stays closed on every insulin atom.

This run tests single-atom norepinephrine END bundles with and without attributes and with or without unit rows. It also identifies insulin's ids from single-atom insulin START bundles. Verified facts go to the notebook first.

```python
with open('/work/notes.md','a') as f:
    f.write("- Run 236 VERIFIED encoder: XALL row = STALL.row (NOT index); prev_row = XALL row. NE START atom = code[9]+kind[0]+ekind[2]+bins.rate[b]+bins.dose[b]+bins.value[0]+numeric@[0,0,zd,1,zr,1]+bias+unit[4]+rate_unit[3]; u_k=LN(mlp2(sum mlp1/sqrt n + size(log(1+n)))); e_k(action-only)=empty-set enc (size(0)); z_post([ctx,e,u]) matches XALL z to 5e-4.\n")
def avec2(code, kind, rate, dose, drug, u, ru, attr=True, units=True):
    x = EM['code'][code] + EM['kind'][kind] + EM['ekind'][2] + EM['bins.value'][0] + NB
    if attr: x = x + EM['bins.rate'][fb_(drug,'rate',rate)] + EM['bins.dose'][fb_(drug,'dose',dose)] + NW@np.array([0,0,zs_(drug,'dose',dose),1,zs_(drug,'rate',rate),1])
    else: x = x + EM['bins.rate'][0] + EM['bins.dose'][0]
    if units: x = x + EM['categorical.unit'][u] + EM['categorical.rate_unit'][ru]
    return x
def zerr(items, f):
    return float(np.mean([np.abs(zpo(c, e_empty, uenc([f(rt,ds)]))-XALL[jx,960:992]).mean() for (jx,c,rt,ds) in items]))
def items_of(df_, n=30):
    it = []
    for r in df_.itertuples():
        if cnt.get((r.stay_id, r.m),0)!=1 or (int(r.stay_id),int(r.m)) not in SI: continue
        j = SI[(int(r.stay_id),int(r.m))]; p = STALL.prev_row.values[j]; dt = float((STALL.minute.values[j]-STALL.prev_min.values[j])*60)
        if np.isnan(p) or dt<=0: continue
        it.append((int(xr[j]), ctx_of(int(p), dt), r.rate, r.dose))
        if len(it)>=n: break
    return it
itE = items_of(EE[EE.event=='norepinephrine']); print('NE END items', len(itE))
for nm_, f_ in [('attr+units', lambda rt,ds: avec2(9,2,rt,ds,'norepinephrine',4,3)), ('noattr+units', lambda rt,ds: avec2(9,2,rt,ds,'norepinephrine',4,3,attr=False)),
                ('attr-nounits', lambda rt,ds: avec2(9,2,rt,ds,'norepinephrine',4,3,units=False)), ('noattr-nounits', lambda rt,ds: avec2(9,2,rt,ds,'norepinephrine',4,3,attr=False,units=False))]:
    print(nm_, round(zerr(itE, f_),4))
itI = items_of(T4[T4.event=='insulin']); print('INS START items', len(itI))
rI = sorted((zerr(itI, lambda rt,ds,u=u,ru=ru: avec2(8,0,rt,ds,'insulin',u,ru)), u, ru) for u in range(6) for ru in range(4)); print([(round(a,4),b,c) for a,b,c in rI[:3]])
```