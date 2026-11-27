Exact for glucose now: finger glucose gets no unit embedding at all, and lab glucose uses unit row 5. So the unit rows follow the source data's unit strings, not the vocab's code units. That also casts doubt on the dose-unit rows I picked for the drugs (mg 4, units 5), which left a small 0.008 residual. I'll recheck them over rows 3–5 and "none", now that row 5 is known to belong to lab glucose. An exact zero for some combination would make the whole encoder and z-posterior replica exact.

```python
def atom_e_full(mg_row,units_row):
    base_f=atom_e_side('left')
    def f(a,umap,rumap,end_attr=False):
        um2={'mmHg':2}
        if a['ev']=='glucose_lab': um2['mg/dL']=5
        if mg_row is not None: um2['mg']=mg_row
        if units_row is not None: um2['units']=units_row
        return base_f(a,um2,rumap,end_attr)
    return f
res=[]
for mr in [None,3,4,5]:
    for ur in [None,3,4,5]:
        atom_e=atom_e_full(mr,ur)
        res.append((zpost_batch4({},RM,types={'A'},end_attr=False,end_in_action=False,nst=20),mr,ur))
res.sort(key=lambda r:r[0]); print(res[:5])
```