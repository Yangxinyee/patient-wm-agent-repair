A ctx built without the time-flow decay helps (0.055 against 0.077), but nothing matches yet. The block order is confirmed as [ctx, e, u], since the other orders are far worse, and gap must take log(dt)/10. The remaining error has to come from e_k, or from how the decay enters ctx.

Next I will break the error down by bundle type: number of atoms, which codes are present, and the hour. I will do this for both ctx variants, on all measurement-only bundles. If single-atom bundles match well and multi-atom bundles do not, the set aggregation is wrong. If the error is uniform, the atom embedding itself is off. If the error is near zero when no time has passed, the decay handling is the issue.

```python
ctx_all_d = ctx_rows(only_meas)
pvA = torch.tensor(SB.prev.values[only_meas].astype(np.int64)); dtsA = torch.tensor(SB.dts.values[only_meas], dtype=torch.float32)
ctx_all_nd = OUT[pvA] + ln(HS[pvA].mean(1), 'core.out_norm') + mlp((torch.log(dtsA)/10)[:, None], 'gap', F.gelu)
subA = AT[AT.row.isin(only_meas)].copy(); rmap = {r: i for i, r in enumerate(only_meas)}
grpA = torch.tensor(subA.row.map(rmap).values)
cA = torch.tensor(subA.event.map(cid).values); vA = subA.value.values; zvA = np.zeros(len(subA)); vbA = np.zeros(len(subA), np.int64)
for ev in codes_names:
    m = (subA.event == ev).values
    if m.any(): zvA[m] = zsc(ev, 'value', vA[m]); vbA[m] = fbin(ev, 'value', vA[m])
uA = torch.tensor(subA.event.map(unit_best).values)
numfA = torch.zeros(len(subA), 6); numfA[:, 0] = torch.tensor(zvA, dtype=torch.float32); numfA[:, 1] = 1
avA = (P['embed.code.weight'][cA] + P['embed.kind.weight'][0] + P['embed.ekind.weight'][1] + P['embed.kq.weight'][0]
       + P['embed.categorical.unit.weight'][uA] + P['embed.bins.value.weight'][torch.tensor(vbA)] + lin(numfA, 'embed.numeric'))
eA = ds_act(avA, grpA, len(only_meas), 'encode', F.gelu)
ztA = ZP[torch.tensor(only_meas)]
errs = {}
for nm_, cc_, ee_ in [('d', ctx_all_d, eA), ('nd', ctx_all_nd, eA), ('nd_e0', ctx_all_nd, torch.zeros_like(eA)), ('d_e0', ctx_all_d, torch.zeros_like(eA))]:
    errs[nm_] = (lin(F.gelu(lin(torch.cat([cc_, ee_, torch.zeros_like(ee_)], 1), 'z_post.0')), 'z_post.2')[:, :32] - ztA).abs().mean(1).numpy()
G = pd.DataFrame({'row': only_meas, 'n': SB.n_atoms.values[only_meas], 'hour': hrs_S[only_meas], 'dts': SB.dts.values[only_meas], **errs})
G['codes'] = subA.groupby('row').event.apply(lambda s: ','.join(sorted(x[:5] + x[-3:] for x in s))).reindex(only_meas).values
print(G.groupby('n')[['d', 'nd', 'nd_e0', 'd_e0']].mean().round(4).to_string())
print(G.groupby('codes')[['d', 'nd', 'nd_e0', 'd_e0']].agg(['mean']).round(4).to_string())
print(G.groupby(G.hour.clip(-1, 3))[['d', 'nd']].mean().round(4).to_string())
print('ztA mean abs', ztA.abs().mean().item())
```