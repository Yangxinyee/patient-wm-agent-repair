Run 83 shows that the unit index, the bin side and the z-score scaling barely matter. No choice brings the single-reading bundles below about 0.5. So the per-field lookups are not the main fault.

The difference between visit-only bundles (error 0.19) and reading bundles (0.5 and more) must then come from the one part that only readings use: the numeric features fed to embed.numeric. The documented column order is [z_value, has_value, z_dose, has_dose, z_rate, has_rate]. If the real order differs, every reading would get a wrong numeric vector, which matches what I see.

I'll test every placement of the value's z-score and has-flag among the 6 columns, on map_arterial-only and glucose_finger-only bundles. A correct placement should drop the error sharply for both codes.

```python
def atom_vec3(a, zc, hc, side='left', unit_idx=0):
    c = a['code']; v = T_['embed.code.weight'][code_id[c]] + T_['embed.kind.weight'][a['kind']] + T_['embed.ekind.weight'][code_ek[c]] + T_['embed.kq.weight'][0]
    if unit_idx: v = v + T_['embed.categorical.unit.weight'][unit_idx]
    num = [0.0]*6; d_ = numtab[(c,'value')]
    num[zc] = float(np.clip((a['value']-d_['median'])/d_['iqr'], -10, 10)); num[hc] = 1.0
    v = v + T_['embed.bins.value.weight'][fbin(c,'value',a['value'],side)]
    return v + T_['embed.numeric.weight'] @ torch.tensor(num) + T_['embed.numeric.bias']
for code in ['map_arterial', 'glucose_finger']:
    rs = []
    for zc in range(6):
        for hc in range(6):
            if zc == hc: continue
            rs.append((single_err(code, lambda a, zc=zc, hc=hc: atom_vec3(a, zc, hc), max_n=40), zc, hc))
    rs.sort(key=lambda t: t[0][0]); print(code, rs[:6])
```