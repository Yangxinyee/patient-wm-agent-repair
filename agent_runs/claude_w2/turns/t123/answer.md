Norepinephrine start atoms are now exact with unit row 4 and rate-unit row 3 (error 0.0000 for norepinephrine-only bundles). Every bundle that contains an END atom still carries an error of about 0.0016 to 0.0025. END atoms carry no rate or dose, so they probably also carry no unit or rate unit (row 0) instead of the drug's rows.

Next I test that: I set END drug atoms to unit and rate-unit row 0, fix norepinephrine starts at (4, 3), and scan the insulin unit row (0–5) with rate-unit row 2 or 0. The hypothesis is confirmed if all action categories, including the "+E" ones, reach about 0.0000 for one insulin setting.

```python
Ut=WO['embed.categorical.unit.weight']; Rt=WO['embed.categorical.rate_unit.weight']
def xa_full(um_,rm_,end_unit=0,end_rate=0):
    xx=xa_with(um_,rm_)
    for cid in [8,9]:
        m_=torch.as_tensor((A0n.code.values==cid)&(A0n.kind.values==2))
        xx[m_]+=Ut[end_unit]-Ut[um_[cid]]+Rt[end_rate]-Rt[rm_[cid]]
    return xx
rows_=[]
for iu in range(6):
    for ir in [0,2]:
        um_=dict(um); um_[8]=iu; um_[9]=4; rm_=dict(rmp); rm_[8]=ir; rm_[9]=3
        er=zerr_cat(*encode_from_xa(A0n,xa_full(um_,rm_),WO)); rows_.append(((iu,ir),er))
for k_,v_ in sorted(rows_,key=lambda r: sum(r[1].values()))[:4]: print(k_,{str(a):b for a,b in v_.items()})
```