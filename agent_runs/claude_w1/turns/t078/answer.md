MAP-only bundles reproduce exactly (error 0.000). All of the remaining error comes from glucose atoms: about 0.07–0.15 for finger glucose and 0.45–0.56 for lab glucose. MAP is exact, so the encoder itself is fine and only the glucose atoms' embedding is off. The most likely culprit is the unit row: the two glucose codes may not share a unit row, or mg/dL isn't row 3. I'll search the unit row for finger and lab glucose independently, on bundles containing glucose.

```python
def atom_e_unit(fu,lu):
    base_f=atom_e_side('left')
    def f(a,umap,rumap,end_attr=False):
        um2=dict(umap)
        if a['ev']=='glucose_finger':
            um2={k:v for k,v in umap.items() if k!='mg/dL'}
            if fu is not None: um2['mg/dL']=fu
        if a['ev']=='glucose_lab':
            um2={k:v for k,v in umap.items() if k!='mg/dL'}
            if lu is not None: um2['mg/dL']=lu
        return base_f(a,um2,rumap,end_attr)
    return f
res=[]
for fu in [None,1,3,4,5]:
    for lu in [None,1,3,4,5]:
        atom_e=atom_e_unit(fu,lu)
        zr_=zpost_rows(nst=20)
        gl=zr_.comp.str.contains('glucose')
        res.append((round(zr_.err[gl].mean(),4),fu,lu))
res.sort(key=lambda r:r[0]); print(res[:6])
atom_e=atom_e_side('left')
```