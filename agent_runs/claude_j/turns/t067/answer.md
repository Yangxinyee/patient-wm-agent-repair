Bundles with a single start atom match exactly under both hypotheses, but no multi-atom bundle does. So something specific to END atoms or to combining several atoms is still wrong.

I'll separate the two using 2-atom bundles. I'll try three ways of combining the atoms (sum/√n as documented, the mean, a plain sum). For the END atom I'll try other kq rows and other event-kind values, with and without the infusion's rate and dose. An exact match would pin down both pieces. If only END-free combinations work, the problem is purely in the END embedding.

```python
def encode_v(atoms, p, agg='sqrt'):
    A = torch.stack(atoms); n = len(atoms)
    s1 = mlp(A, p+'.mlp1').sum(0)
    s1 = {'sqrt': s1/math.sqrt(n), 'mean': s1/n, 'sum': s1}[agg]
    return ln(mlp(s1 + lin(torch.tensor([math.log(1+n)]), p+'.size'), p+'.mlp2'), p+'.norm')
def drug_atom2(ev, kind, rate, dose, end_attrs=True, end_kq=0, end_ekind=2, end_kind=2):
    ui, ri = CFG[ev]
    if kind=='start':
        return drug_atom(ev, kind, rate, dose, ui, ri)
    x = Wt['embed.code.weight'][code_id[ev]] + Wt['embed.kind.weight'][end_kind] + Wt['embed.ekind.weight'][end_ekind] + Wt['embed.kq.weight'][end_kq]
    x = x + Wt['embed.categorical.unit.weight'][ui] + Wt['embed.categorical.rate_unit.weight'][ri]
    if end_attrs:
        x = x + Wt['embed.bins.dose.weight'][fbin(ev,'dose',dose)] + Wt['embed.bins.rate.weight'][fbin(ev,'rate',rate)]
        num = torch.tensor([0.,0., zsc(ev,'dose',dose), 1., zsc(ev,'rate',rate), 1.])
    else: num = torch.zeros(6)
    return x + Wt['embed.numeric.weight'] @ num + Wt['embed.numeric.bias']
def a_err2(tlist, agg='sqrt', **kw):
    ee=[]
    for r_, rp, dt, ats in tlist:
        atoms = [drug_atom2(ev, kd, rt, ds, **kw) for ev,kd,rt,ds in ats]
        u_ = encode_v(atoms, 'encode_action', agg)
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([math.log(dt)/10]), 'gap')
        zp = mlp(torch.cat([ctx, e_empty, u_]), 'z_post')[:32]
        ee.append((zp - Xs[r_, 960:992]).abs().max().item())
    return round(float(np.median(ee)),6)
t2 = [a for a in atests if len(a[3])==2][:15]
print(t2[0][3])
for agg in ['sqrt','mean','sum']:
    for ea in [True, False]:
        print(agg, ea, a_err2(t2, agg, end_attrs=ea), [a_err2(t2, agg, end_attrs=ea, end_kq=q) for q in [1,2,3]])
```