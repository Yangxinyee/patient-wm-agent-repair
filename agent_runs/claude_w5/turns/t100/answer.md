Run 95 solves most of the rebuild. With the empty-set encoding for u, the documented time flow, gap(log(dt in seconds)/10), and unit indices glucose→3 and MAP→2, the relative error of the z posterior falls from about 0.5 to 0.051. The unit order (mmHg = 2, mg/dL = 3) suggests the unit vocabulary was built in order of first appearance in the data, so the two drug units are probably 4 and 5 in some order.

Next I'll find where the remaining 5% sits by grouping bundles by atom composition with this configuration. Then I'll extend the check to bundles with drugs. There I'll test the unit and rate_unit indices for insulin and norepinephrine, and an empty-set e for drug-only bundles. If the drug bundles also reach about 5% error or better, I can move on to checking the full step: prelude, recurrent block and coda.

```python
optB = {'unit': {'glucose_finger':3, 'glucose_lab':3, 'map_arterial':2, 'map_cuff':2}, 'rate_unit': {}, 'side': 'left'}
E_empty = setenc_empty('encode')
def zp_full(opt, stays, kind='meas', empty_e='enc', max_n=300, by_comp=False):
    errs = defaultdict(list) if by_comp else []
    for s_ in stays:
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        for k_ in range(1, len(bl_)):
            atoms = bl_[k_][1]
            wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
            if kind == 'meas' and (aa or not wa): continue
            if kind == 'drug' and not aa: continue
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
            ctx = out_p + ln_(flow_c(h_p, dt, 1.0).mean(0),'core.out_norm') + G_in(dt, 's10')
            e = setenc_a([atom_vec(a, opt) for a in wa], 'encode', Fnn.gelu) if wa else (E_empty if empty_e == 'enc' else torch.zeros(192))
            u = setenc_a([atom_vec(a, opt) for a in aa], 'encode_action', Fnn.gelu) if aa else U_empty
            zh = mlp_(torch.cat([ctx, e, u]), 'z_post', 'gelu')[:32]; zt = Xt[r0_+k_, 960:]
            er = float(torch.norm(zh-zt)/torch.norm(zt))
            if by_comp: errs['+'.join(sorted(set(a['code']+str(a['kind']) for a in atoms)))].append(er)
            else: errs.append(er)
            if not by_comp and len(errs) >= max_n: break
        if not by_comp and len(errs) >= max_n: break
    if by_comp: return sorted([(k_, len(v_), round(float(np.median(v_)),4)) for k_, v_ in errs.items()], key=lambda t: -t[1])[:12]
    return round(float(np.median(errs)),4), round(float(np.mean(errs)),4), len(errs)
print('meas by comp:', zp_full(optB, list(bundles_B.keys())[:20], 'meas', by_comp=True))
rd = []
for iu, nu in [(4,5),(5,4)]:
    for ir, nr_ in [(2,3),(3,2)]:
        for emp in ['enc','zero']:
            o_ = {'unit': dict(optB['unit'], insulin=iu, norepinephrine=nu), 'rate_unit': {'insulin': ir, 'norepinephrine': nr_}, 'side': 'left'}
            rd.append((zp_full(o_, list(bundles_B.keys())[:10], 'drug', empty_e=emp, max_n=120), iu, nu, ir, nr_, emp))
rd.sort(key=lambda t: t[0][0]); print('drug:', rd[:5])
```