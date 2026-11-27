Sending END atoms to the measurement encoder (e_k) without their rate and dose cuts the error to about 0.007–0.01. That is close, but not the ~1e-7 of the other matches. So the routing is probably right and a detail of the END atom's embedding is still off.

I'll search its remaining options in that routing: event-kind row 0–4, kq row 0–3, and with or without the dose-unit and rate-unit rows. An exact hit would complete the embedding for every atom type in the data, apart from the admission records.

```python
def drug_end(ev, end_kq=0, end_ekind=2, use_unit=True, use_rate_unit=True):
    ui, ri = CFG[ev]
    x = Wt['embed.code.weight'][code_id[ev]] + Wt['embed.kind.weight'][2] + Wt['embed.ekind.weight'][end_ekind] + Wt['embed.kq.weight'][end_kq]
    if use_unit: x = x + Wt['embed.categorical.unit.weight'][ui]
    if use_rate_unit: x = x + Wt['embed.categorical.rate_unit.weight'][ri]
    return x + Wt['embed.numeric.bias']
def az_err_route2(tlist, **kw):
    ee=[]
    for r_, rp, dt, ats in tlist:
        starts = [drug_atom2(ev, kd, rt, ds) for ev,kd,rt,ds in ats if kd=='start']
        ends = [drug_end(ev, **kw) for ev,kd,rt,ds in ats if kd=='end']
        u_ = encode(starts, 'encode_action') if starts else u_empty
        e_ = encode(ends, 'encode') if ends else e_empty
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([math.log(dt)/10]), 'gap')
        zp = mlp(torch.cat([ctx, e_, u_]), 'z_post')[:32]
        ee.append((zp - Xs[r_, 960:992]).abs().max().item())
    return round(float(np.median(ee)),7)
tl2 = [a for a in atests if len(a[3])==2][:12]
res_ = []
for ek in range(5):
    for q in range(4):
        for uu in [True, False]:
            for rr in [True, False]:
                res_.append((az_err_route2(tl2, end_kq=q, end_ekind=ek, use_unit=uu, use_rate_unit=rr), ek, q, uu, rr))
print(sorted(res_)[:6])
```