The decoding test shows the existing state carries little dose information. The within-tier deviation of the last insulin rate is decodable with a cross-validated R² of only 0.27, and the earlier lags hardly at all: 0.07, 0.11 and about 0. A head-only readout would therefore be weak, and I cannot predict how it extrapolates to doubled doses.

The records gave two useful results:
- The tier mean rates are 1.0, 2.0, 3.0 and 3.7 units/h.
- With past glucose included, the direct effect of a rate deviation is −2.1, −1.54 and −0.53 at lags 1 to 3, then turns slightly positive. The glucose lags carry persistence of 0.46, 0.28 and 0.18. In a rollout, a lag-1 shock carried forward by this persistence averages to about −8.4 × r over a 12-hour window. The true effect gives about −8.7 × r, so a lag-1-led effect could reproduce the readout if the model propagates shifts the way the data do.

To inject the dose properly I still need the encoders. The last numeric search favoured z_value in column 0 and has_value in column 3, which suggests the layout [z_value, z_dose, z_rate, has_value, has_dose, has_rate]. The error stayed large, so something structural is still off.

One untested possibility is that the stored z is the prior mean plus the z_post output (a residual posterior). The other is that ctx uses the undecayed h. Next I will test both with the new numeric layout. A large drop in error for one form would identify the posterior parameterisation. If nothing improves, the error lies inside the encoder itself.

```python
def enc_meas(subA_, rows_, unit_map, cz=0, ch=3, add_kq=1, add_kind=1, A=F.gelu):
    rm_ = {r: i for i, r in enumerate(rows_)}; g_ = torch.tensor(subA_.row.map(rm_).values)
    c_ = torch.tensor(subA_.event.map(cid).values); v_ = subA_.value.values; zv_ = np.zeros(len(subA_)); vb_ = np.zeros(len(subA_), np.int64)
    for ev in codes_names:
        m = (subA_.event == ev).values
        if m.any(): zv_[m] = zsc(ev, 'value', v_[m]); vb_[m] = fbin(ev, 'value', v_[m])
    nf = torch.zeros(len(subA_), 6); nf[:, cz] = torch.tensor(zv_, dtype=torch.float32); nf[:, ch] = 1
    av = (P['embed.code.weight'][c_] + add_kind*P['embed.kind.weight'][0] + P['embed.ekind.weight'][1] + add_kq*P['embed.kq.weight'][0]
          + P['embed.categorical.unit.weight'][torch.tensor(subA_.event.map(unit_map).values)] + P['embed.bins.value.weight'][torch.tensor(vb_)] + lin(nf, 'embed.numeric'))
    return ds_act(av, g_, len(rows_), 'encode', A)
rows_t = only_meas[:600]; subT = AT[AT.row.isin(rows_t)].copy(); zT = ZP[torch.tensor(rows_t)]
pvT = torch.tensor(SB.prev.values[rows_t].astype(np.int64)); dtsT = torch.tensor(SB.dts.values[rows_t], dtype=torch.float32)
gT = mlp((torch.log(dtsT)/10)[:, None], 'gap', F.gelu)
cT_d = OUT[pvT] + ln((HS[pvT]*torch.exp(-dtsT[:, None, None]*torch.exp(P['core.flow.a']))).mean(1), 'core.out_norm') + gT
cT_nd = OUT[pvT] + ln(HS[pvT].mean(1), 'core.out_norm') + gT
for cnm, cT in [('decay', cT_d), ('nodecay', cT_nd)]:
    prT = mlp(cT, 'z_prior', F.gelu)[:, :32]
    print(cnm, 'prior-mean mae', round((prT - zT).abs().mean().item(), 4))
    best_ = []
    for ug, um in itertools.permutations([2, 3, 4, 5], 2):
        umap = {'glucose_finger': ug, 'glucose_lab': ug, 'map_arterial': um, 'map_cuff': um}
        for add_kq, add_kind in [(1, 1), (0, 0), (1, 0), (0, 1)]:
            eT = enc_meas(subT, rows_t, umap, 0, 3, add_kq, add_kind)
            po = mlp(torch.cat([cT, eT, torch.zeros_like(eT)], 1), 'z_post', F.gelu)[:, :32]
            best_.append((round((po - zT).abs().mean().item(), 4), round((prT + po - zT).abs().mean().item(), 4), ug, um, add_kq, add_kind))
    best_.sort(); print(' plain best', best_[:3]); best_.sort(key=lambda r: r[1]); print(' prior+post best', best_[:3])
```