Run 99 pinpoints the remaining errors:
- **Insulin start records:** a bundle with only a new insulin record now matches exactly, so that atom is right with unit 3 and rate_unit 2.
- **Norepinephrine start records:** lone records are close (about 6% off), so their unit or rate_unit index is slightly off.
- **END atoms:** bundles with only an END atom are badly wrong (78–90%). Either END atoms are routed to the world encoder rather than the action encoder, or they carry different attributes.

I'll test both questions:
- for lone norepinephrine starts, the unit and rate_unit indices;
- for lone END atoms, routing to e versus u, with and without rate and dose, and with and without the unit fields.

Exact matches would complete the encoder.

```python
def zp_custom(stays, filt, route_end, opt, max_n=60):
    errs = []
    for s_ in stays:
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        for k_ in range(1, len(bl_)):
            atoms = bl_[k_][1]
            if not filt(atoms): continue
            wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
            if route_end == 'world':
                wa = wa + [a for a in aa if a['kind'] == 2]; aa = [a for a in aa if a['kind'] != 2]
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
            ctx = out_p + ln_(flow_c(h_p, dt, 1.0).mean(0),'core.out_norm') + G_in(dt, 's10')
            e = setenc_a([atom_vec(a, opt) for a in wa], 'encode', Fnn.gelu) if wa else E_empty
            u = setenc_a([atom_vec(a, opt) for a in aa], 'encode_action', Fnn.gelu) if aa else U_empty
            zh = mlp_(torch.cat([ctx, e, u]), 'z_post', 'gelu')[:32]; zt = Xt[r0_+k_, 960:]
            errs.append(float(torch.norm(zh-zt)/torch.norm(zt)))
            if len(errs) >= max_n: break
        if len(errs) >= max_n: break
    return (round(float(np.median(errs)),4), len(errs)) if errs else None
stays40 = list(bundles_B.keys())[:100]
only_ne0 = lambda atoms: len(atoms) == 1 and atoms[0]['code'] == 'norepinephrine' and atoms[0]['kind'] == 0
rn = []
for nu in [0,3,4,5]:
    for nr_ in [0,2,3]:
        o_ = {'unit': dict(unit_meas, insulin=3, norepinephrine=nu), 'rate_unit': {'insulin': 2, 'norepinephrine': nr_}, 'side': 'left', 'end_attrs': False}
        rn.append((zp_custom(stays40, only_ne0, 'action', o_), nu, nr_))
rn.sort(key=lambda t: t[0][0]); print('ne0:', rn[:4])
only_end = lambda atoms: len(atoms) == 1 and atoms[0]['code'] in ACTION_CODES and atoms[0]['kind'] == 2
re2 = []
for route in ['action','world']:
    for ea in [True, False]:
        for uu in ['with_units','no_units']:
            um = dict(unit_meas, insulin=3, norepinephrine=rn[0][1]) if uu == 'with_units' else dict(unit_meas)
            rr_ = {'insulin': 2, 'norepinephrine': rn[0][2]} if uu == 'with_units' else {}
            o_ = {'unit': um, 'rate_unit': rr_, 'side': 'left', 'end_attrs': ea}
            re2.append((zp_custom(stays40, only_end, route, o_), route, ea, uu))
re2.sort(key=lambda t: t[0][0]); print('end:', re2[:5])
```