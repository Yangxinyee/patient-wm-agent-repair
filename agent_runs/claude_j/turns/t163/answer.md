The visit events include names like `icu_admission`, which my prefix test missed. I'll route only the four measurement codes to `meas_atom`, send every other non-drug event to `visit_atom`, and rerun the one-step check.

```python
MEAS=('glucose_finger','glucose_lab','map_arterial','map_cuff')
def bvec2(atoms, end_attrs=False):
    ua=[];ea=[]
    for ev_,kd_,r_ in atoms:
        if ev_ in ('insulin','norepinephrine'): ua.append(drug_atom(ev_,kd_,r_.rate,r_.dose,*CFG[ev_],attrs_on_end=end_attrs))
        elif ev_ in MEAS: ea.append(meas_atom(ev_,r_.value))
        else: ea.append(visit_atom(ev_,kd_))
    e_=encode(ea,'encode') if ea else e_empty; u_=encode(ua,'encode_action') if ua else u_empty
    return e_,u_
for ea_ in [False,True]:
    try: print('end->u attrs',ea_, [my_one3(st_,k_,ea_) for k_ in range(4,13)])
    except Exception as ex_: print('err', type(ex_).__name__, str(ex_)[:150])
```