`drug_end` takes the drug name, not the event tuple. Removing the END atom from the action set makes bundle 5 much worse (16.7 vs 0.9), so END atoms do enter the action encoding. They are probably just encoded slightly differently from my current version, for example by carrying the ended infusion's rate and dose. Next I'll build END atoms with `drug_atom2` under several settings (attributes on/off, kind and event-kind variants) and score bundle 5 for each. I'll also try sending the END atom to the measurement encoder. The variant that reaches about 1e-5 settles it.

```python
st_atom = None
row_end = ends[0][2]; row_st = starts[0][2]
res_ = []
for ea in [True, False]:
    for ek in [0,1,2,3,4]:
        for kd in [2,0,1]:
            for kq in [0,1,2,3]:
                try:
                    a_end = torch.as_tensor(drug_atom2('norepinephrine','end', row_end.rate, row_end.dose, end_attrs=ea, end_kq=kq, end_ekind=ek, end_kind=kd)).double()
                    a_st = torch.as_tensor(drug_atom2('norepinephrine','start', row_st.rate, row_st.dose)).double()
                    u5 = torch.as_tensor(encode(torch.stack([a_st,a_end]).float(), p='encode_action')).double()
                    res_.append((err_k(st0,5,{5:(e_emp,u5)}), ea, ek, kd, kq))
                except Exception as ex_:
                    res_.append((999, ea, ek, kd, kq, repr(ex_)[:80]))
res_.sort(key=lambda t: t[0]); print(res_[:8])
# sanity: does this construction reproduce the current (bundle_vecs) u?
print('bvd u vs construct (attrs False, ek2,kd2,kq0):', [r for r in res_ if r[1:5]==(False,2,2,0)])
```