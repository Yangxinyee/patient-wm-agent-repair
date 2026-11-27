`encode` expects a list of atom tensors, not a stacked tensor. I'll pass a list and rerun the same grid of END-atom encodings.

```python
res_ = []
for ea in [True, False]:
    for ek in [0,1,2,3,4]:
        for kd in [2,0,1]:
            for kq in [0,1,2,3]:
                try:
                    a_end = drug_atom2('norepinephrine','end', row_end.rate, row_end.dose, end_attrs=ea, end_kq=kq, end_ekind=ek, end_kind=kd)
                    a_st = drug_atom2('norepinephrine','start', row_st.rate, row_st.dose)
                    u5 = torch.as_tensor(encode([a_st,a_end], p='encode_action')).double()
                    res_.append((err_k(st0,5,{5:(e_emp,u5)}), ea, ek, kd, kq))
                except Exception as ex_:
                    res_.append((999, ea, ek, kd, kq, repr(ex_)[:100]))
res_.sort(key=lambda t: t[0]); print(res_[:8])
print('construct (attrs False, ek2,kd2,kq0):', [r for r in res_ if r[1:5]==(False,2,2,0)])
```