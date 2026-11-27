`drug_atom` expects the unit indices as two separate arguments (`unit_ix, rate_unit_ix`), and it has its own `attrs_on_end` switch. I'll unpack `CFG[ev]` and let `attrs_on_end` decide whether end atoms keep their rate and dose. Then I'll rerun the one-step check over bundles 4–12, still looking for errors around 1e-4 everywhere.

```python
def bvec2(atoms, end_attrs=False):
    ua=[];ea=[]
    for ev_,kd_,r_ in atoms:
        if ev_ in ('insulin','norepinephrine'):
            ua.append(drug_atom(ev_,kd_,r_.rate,r_.dose,*CFG[ev_],attrs_on_end=end_attrs))
        elif ev_.startswith('hospital'): ea.append(visit_atom(ev_,kd_))
        else: ea.append(meas_atom(ev_,r_.value))
    e_=encode(ea,'encode') if ea else e_empty; u_=encode(ua,'encode_action') if ua else u_empty
    return e_,u_
for ea_ in [False,True]:
    try: print('end->u attrs',ea_, [my_one3(st_,k_,ea_) for k_ in range(4,13)])
    except Exception as ex_: print('err', type(ex_).__name__, str(ex_)[:150])
```