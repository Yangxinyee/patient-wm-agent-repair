Glucose_lab matches exactly with unit row 5, so the measurement embedding is complete. The unit rows are: arterial and cuff MAP → 2, glucose_finger → 0 (none), glucose_lab → 5, all with kq row 0.

Next are the action atoms. I'll take bundles holding a single drug start atom and no measurements, so the measurement vector is the encoder's empty-set output. For each drug I'll search the dose-unit row × rate-unit row combinations: drug-admin event kind, bins and z-scores for both dose and rate. An exact hit for insulin and norepinephrine would fix the start-atom embedding. The 2-atom start+END bundles come next.

```python
UNIT = {'mmHg':2, 'mg/dL':0}
def meas_atom(code, val):
    ui = {'map_arterial':2, 'map_cuff':2, 'glucose_finger':0, 'glucose_lab':5}[code]
    x = Wt['embed.code.weight'][code_id[code]] + Wt['embed.kind.weight'][0] + Wt['embed.ekind.weight'][1] + Wt['embed.kq.weight'][0]
    x = x + Wt['embed.categorical.unit.weight'][ui] + Wt['embed.bins.value.weight'][fbin(code,'value',val)]
    return x + Wt['embed.numeric.weight'] @ torch.tensor([zsc(code,'value',val), 1., 0., 0., 0., 0.]) + Wt['embed.numeric.bias']
atests = []
row = 0
for s_ in stays_u[:30]:
    bb = build_bundles(tu[tu.stay_id==s_]); keys = sorted(bb.keys())
    for i_, k_ in enumerate(keys):
        evs = bb[k_]
        if i_>0 and all(e2 in ('insulin','norepinephrine') for e2,_,_ in evs):
            atests.append((row+i_, row+i_-1, (k_-keys[i_-1])*60.0, [(e2,k2,r2.rate,r2.dose) for e2,k2,r2 in evs]))
    row += len(keys)
print(len(atests), pd.Series([len(a[3]) for a in atests]).value_counts().to_dict())
def drug_atom(ev, kind, rate, dose, unit_ix, rate_unit_ix, kq=0, attrs_on_end=True):
    kid = 0 if kind=='start' else 2
    x = Wt['embed.code.weight'][code_id[ev]] + Wt['embed.kind.weight'][kid] + Wt['embed.ekind.weight'][2] + Wt['embed.kq.weight'][kq]
    x = x + Wt['embed.categorical.unit.weight'][unit_ix] + Wt['embed.categorical.rate_unit.weight'][rate_unit_ix]
    if kind=='start' or attrs_on_end:
        x = x + Wt['embed.bins.dose.weight'][fbin(ev,'dose',dose)] + Wt['embed.bins.rate.weight'][fbin(ev,'rate',rate)]
        num = torch.tensor([0.,0., zsc(ev,'dose',dose), 1., zsc(ev,'rate',rate), 1.])
    else:
        num = torch.zeros(6)
    return x + Wt['embed.numeric.weight'] @ num + Wt['embed.numeric.bias']
def a_err(tlist, cfg):
    ee=[]
    for r_, rp, dt, ats in tlist:
        atoms = [drug_atom(ev, kd, rt, ds, cfg[ev][0], cfg[ev][1], attrs_on_end=cfg.get('end_attrs',True)) for ev,kd,rt,ds in ats]
        u_ = encode(atoms, 'encode_action')
        out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
        dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([math.log(dt)/10]), 'gap')
        zp = mlp(torch.cat([ctx, e_empty, u_]), 'z_post')[:32]
        ee.append((zp - Xs[r_, 960:992]).abs().max().item())
    return round(float(np.median(ee)),6) if ee else None
single = {ev: [a for a in atests if len(a[3])==1 and a[3][0][0]==ev and a[3][0][1]=='start'] for ev in ['insulin','norepinephrine']}
print({k_: len(v_) for k_, v_ in single.items()})
for ev in ['insulin','norepinephrine']:
    best=[]
    for ui in range(6):
        for ri in range(4):
            cfg = {ev:(ui,ri)}
            best.append((a_err(single[ev][:10], cfg), ui, ri))
    print(ev, sorted(best)[:4])
```