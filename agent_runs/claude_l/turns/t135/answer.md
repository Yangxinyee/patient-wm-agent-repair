Swapping the measurement units cuts the median e error on measurement bundles from 1.31 to 0.23. So glucose (mg/dL) uses unit row 5 and MAP (mmHg) row 2, the reverse of my mapping. The drug units are already right, since u matches exactly.

About 0.23 of error remains, so something else is also off. For END atoms in e, dropping either the numerics or the bins helps (0.61 → 0.32 or 0.40), so they may carry neither, or carry them differently.

Next I'll test combinations:
- unit swap together with the tie rule, value-bin offset and z variants for measurements;
- END-atom variants: no numerics and no bins, plus unit and rate-unit options.

```python
def comp(*fs):
    def f(a):
        for g in fs: a=g(a)
        return a
    return f
def Vnoclip(a):
    if a['ekind']==1:
        nm,x=raw_val(a); d=NUMV[(nm,'value')]; n=list(a['num']); n[0]=(x-d['median'])/d['iqr']; a['num']=tuple(n)
    return a
def Vvbp1(a):
    if a['ekind']==1: a['vb']=min(a['vb']+1,16)
    return a
for nm,v in [('unit',Vunit),('unit+right',comp(Vunit,Vright)),('unit+vb-1',comp(Vunit,Vvbm1)),('unit+vb+1',comp(Vunit,Vvbp1)),('unit+nohas',comp(Vunit,Vnohas))]:
    print(nm,'meas', np.round(e_err(v,idx_meas[:200]),4))
def Vend_bare(a):
    if a['kind']==2 and a['act']: a['num']=(0,0,0,0,0,0); a['db']=0; a['rb']=0
    return a
def Vend_bare_nou(a):
    if a['kind']==2 and a['act']: a['num']=(0,0,0,0,0,0); a['db']=0; a['rb']=0; a['unit']=0; a['rate_unit']=0
    return a
def Vend_ratefeat(a):
    if a['kind']==2 and a['act']: n=list(a['num']); a['num']=(0,0,0,0,n[4],n[5]); a['db']=0
    return a
for nm,v in [('end_bare',Vend_bare),('end_bare_nounit',Vend_bare_nou),('end_rateonly',Vend_ratefeat)]:
    print(nm,'end', np.round(e_err(v,idx_end),4))
```