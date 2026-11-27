My bin rule (`searchsorted`, side='left') already counts strictly-lower edges, as the vocab note says. The tie rule is still worth testing both ways, alongside several other variants.

I'll build a harness that compares my replica's e projections with the probe's true projections. It routes measurement, visit and END atoms to e, and runs over about 400 measurement bundles plus the END-only bundles. Variants:
- ties counted the other way;
- unit rows swapped;
- the order of the numeric features changed;
- value bin offset by one;
- has_value set to zero;
- for END atoms, dropping the dose/rate features.

The variant that brings the error to about 1e-6 identifies the bug.

```python
idx_meas=[i for i,r in enumerate(recs) if any((not a['act']) and a['ekind']==1 for a in BUN[r[0]][r[1]][1])][:400]
idx_end=[i for i,r in enumerate(recs) if any(a['kind']==2 and a['act'] for a in BUN[r[0]][r[1]][1]) and not any((not a['act']) for a in BUN[r[0]][r[1]][1])][:150]
def raw_val(a):
    nm=[k for k,v in CODE_ID.items() if v==a['code']][0]; d=NUMV[(nm,'value')]; return nm, a['num'][0]*d['iqr']+d['median']
def e_err(var, idxs):
    errs=[]
    for i in idxs:
        s,k=recs[i][0],recs[i][1]; atoms=BUN[s][k][1]
        ea=[var(dict(a)) for a in atoms if (not a['act']) or a['kind']==2]
        e=set_enc(W0, ea, 'encode').detach().numpy()
        errs.append(np.abs(Pe@e-recs[i][4]).max())
    return np.median(errs), np.percentile(errs,90)
def V0(a): return a
def Vright(a):
    if a['ekind']==1:
        nm,x=raw_val(a); a['vb']=1+int(np.searchsorted(np.array(NUMV[(nm,'value')]['fine_bin_edges']),x,side='right'))
    return a
def Vunit(a):
    if a['ekind']==1: a['unit']={2:5,5:2}[a['unit']]
    return a
def Vnumswap(a):
    if a['ekind']==1: n=list(a['num']); a['num']=(n[1],n[0])+tuple(n[2:])
    return a
def Vvbm1(a):
    if a['ekind']==1: a['vb']=max(a['vb']-1,0)
    return a
def Vnohas(a):
    if a['ekind']==1: n=list(a['num']); n[1]=0; a['num']=tuple(n)
    return a
def Vend_nonum(a):
    if a['kind']==2 and a['act']: a['num']=(0,0,0,0,0,0)
    return a
def Vend_nobins(a):
    if a['kind']==2 and a['act']: a['db']=0; a['rb']=0
    return a
for nm,v in [('V0',V0),('right',Vright),('unit',Vunit),('numswap',Vnumswap),('vb-1',Vvbm1),('nohas',Vnohas)]:
    print(nm,'meas', np.round(e_err(v,idx_meas[:200]),4))
for nm,v in [('V0',V0),('end_nonum',Vend_nonum),('end_nobins',Vend_nobins)]:
    print(nm,'end', np.round(e_err(v,idx_end),4))
```