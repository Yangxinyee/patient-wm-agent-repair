No combining rule or END variant gives an exact match for start + END bundles. So the END atom may not belong to the action set at all. The doc only says insulin and norepinephrine atoms are actions. An END atom might instead be encoded with the measurement and visit events into e_k, leaving only start atoms in u_k.

I'll test that routing, with and without the END atom's rate and dose, on bundles of 2, 3 and 4 atoms. An exact match would complete the embedding. It would also mean the dose edit's doubled END atoms reach the state through e_k, the vector stored in the model's memory.

```python
def az_err_route(tlist, end_attrs=True, end_kq=0, end_ekind=2):
    ee=[]
    for r_, rp, dt, ats in tlist:
        starts = [drug_atom2(ev, kd, rt, ds) for ev,kd,rt,ds in ats if kd=='start']
        ends = [drug_atom2(ev, kd, rt, ds, end_attrs=end_attrs, end_kq=end_kq, end_ekind=end_ekind) for ev,kd,rt,ds in ats if kd=='end']
        u_ = encode(starts, 'encode_action') if starts else u_empty
        e_ = encode(ends, 'encode') if ends else e_empty
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([math.log(dt)/10]), 'gap')
        zp = mlp(torch.cat([ctx, e_, u_]), 'z_post')[:32]
        ee.append((zp - Xs[r_, 960:992]).abs().max().item())
    return round(float(np.median(ee)),6)
for ea in [True, False]:
    print('END->encode, attrs', ea, [az_err_route([a for a in atests if len(a[3])==n_][:15], end_attrs=ea) for n_ in [2,3,4]])
```