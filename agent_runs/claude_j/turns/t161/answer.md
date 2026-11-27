The trace shows the problem. `bundle_vecs` sends infusion-end atoms to the measurement encoder (e), but the architecture says all insulin/norepinephrine atoms are actions and go to `encode_action` (u). I'll write a variant that routes end atoms into u through `drug_atom`, with or without their rate and dose, and repeat the one-step check, keeping the current bundle in memory. If routing ends to u brings errors to about 1e-4 on both drug and reading bundles, the replica is exact again.

```python
print('drug_atom sig', inspect.signature(drug_atom), '| drug_end sig', inspect.signature(drug_end))
def bvec2(atoms, end_attrs=False):
    ua=[];ea=[]
    for ev_,kd_,r_ in atoms:
        if ev_ in ('insulin','norepinephrine'):
            if kd_=='start' or end_attrs: ua.append(drug_atom(ev_,kd_,r_.rate,r_.dose,CFG[ev_]))
            else: ua.append(drug_atom(ev_,kd_,float('nan'),float('nan'),CFG[ev_]))
        elif ev_.startswith('hospital') or kd_ not in ('world',): ea.append(visit_atom(ev_,kd_))
        else: ea.append(meas_atom(ev_,r_.value))
    e_=encode(ea,'encode') if ea else e_empty; u_=encode(ua,'encode_action') if ua else u_empty
    return e_,u_
def my_one3(stay,k,ea_):
    bl_=BUN[stay]; idx_=np.where(sid_a==stay)[0]; Es=[bvec2(b[1],ea_) for b in bl_]
    tms=np.array([b[0]*60.0 for b in bl_]); js=list(range(k+1))[-128:]
    Mem=torch.stack([Es[j][0] for j in js]); ages=torch.tensor([tms[k]-tms[j] for j in js],dtype=torch.float64)
    o_,y_=step(Xz[idx_[k-1],:192], Xz[idx_[k-1],192:960].reshape(4,192), float(tms[k]-tms[k-1]), Es[k][0], Es[k][1], Xz[idx_[k],960:992], Mem, ages)
    return round(max(float((o_-Xz[idx_[k],:192]).abs().max()), float((y_.reshape(-1)-Xz[idx_[k],192:960]).abs().max())),4)
for ea_ in [False,True]:
    try: print('end->u attrs',ea_, [my_one3(st_,k_,ea_) for k_ in range(4,13)])
    except Exception as ex_: print('err', type(ex_).__name__, str(ex_)[:150])
```