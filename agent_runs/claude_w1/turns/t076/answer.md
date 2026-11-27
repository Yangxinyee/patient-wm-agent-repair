Drug bundles are now exact to 0.7%, and END-only and visit bundles to 0.0. Reading-only bundles still have 16% error, so the remaining discrepancy is specific to measurement atoms.

The likeliest cause is bin assignment for values that sit exactly on a bin edge. Readings are integers and so are many edges, so this happens often, and "number of edges below x" is ambiguous between counting edges strictly below x and edges at or below it. I'll test both counts on reading-only bundles, for the value bin and also the head's last-value bin.

```python
def fb_(code,attr,x,side='left'): e=nm_[(code,attr)]; return int(1+np.searchsorted(np.array(e['fine_bin_edges']),x,side=side))
_atom_e_orig=atom_e
def atom_e_side(side):
    def f(a,umap,rumap,end_attr=False):
        ev=a['ev']; v=EMB['embed.code.weight'][CID[ev]]+EMB['embed.kind.weight'][a['kind']]+EMB['embed.ekind.weight'][EV2EK[ev]]+EMB['embed.kq.weight'][0]
        num=np.zeros(6); du=a['du'] if isinstance(a['du'],str) else None
        unit=du if du else VOCAB['codes'][CID[ev]].get('unit')
        if a['kind']==0 or end_attr:
            if unit in umap: v=v+EMB['embed.categorical.unit.weight'][umap[unit]]
            if isinstance(a['ru'],str) and a['ru'] in rumap: v=v+EMB['embed.categorical.rate_unit.weight'][rumap[a['ru']]]
        for col,attr,bn,i in [('value','value','embed.bins.value.weight',0),('dose','dose','embed.bins.dose.weight',2),('rate','rate','embed.bins.rate.weight',4)]:
            x=a[col]
            if x is not None and x==x:
                num[i]=zs_(ev,attr,x); num[i+1]=1; v=v+EMB[bn][fb_(ev,attr,x,side)]
        return v+EMB['embed.numeric.weight']@num+EMB['embed.numeric.bias']
    return f
for side in ['left','right']:
    atom_e=atom_e_side(side)
    print(side,'M',zpost_batch4(UM,RM,types={'M'},nst=20),'A',zpost_batch4(UM,RM,types={'A'},nst=20))
```