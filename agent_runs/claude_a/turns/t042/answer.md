The bundle structure matches the states file exactly:
- Readings sit on the hour.
- Drug starts are at h+0.083, sharing a bundle with the END atoms of the previous hour's records.
- `has_insulin` marks an insulin start atom, not an END atom.
- The stay closes with an END-atom bundle at 25.0833, then a 2-atom bundle at 25.1667, probably the visits ending.

The embedding row norms show which categories occur:
- Rows near 0.277 were never trained, so they never occur: kq rows 1 to 3, text, action, route, status and discharged_to.
- kq always uses row 0.
- Rate units use rows 2 and 3; the unit table uses rows 2 to 5.
- Atom kinds 0 (WORLD) and 2 (END) occur.
- The high-insulin dose-bin rows 13 to 16 are barely trained.

Next I will build every bundle's atoms from the records and check my encoder through the z posterior. z_post([ctx, e, u]) has to reproduce the z mean stored in the states file, and ctx comes straight from the previous stored state. I will start with bundles holding only measurements, so u = 0, and search which unit row belongs to mg/dL and which to mmHg. An error of order 1e-4 for one assignment would confirm both the measurement encoder and the z_post input layout. If no assignment matches, something about e_k or u_k for an empty set is still wrong.

```python
# build atoms per bundle for states stays
stays_S = pd.unique(sids_S)
Rs = tr[tr.stay_id.isin(stays_S)].copy()
rows = []
for r in Rs.itertuples(index=False):
    if r.event in ('glucose_finger','glucose_lab','map_arterial','map_cuff'):
        rows.append((r.stay_id, r.hour, r.event, 0, r.value, np.nan))
    elif r.event in ('insulin','norepinephrine'):
        rows.append((r.stay_id, r.hour, r.event, 0, np.nan, r.rate))
        rows.append((r.stay_id, r.end_hour, r.event, 2, np.nan, r.rate))
    else:
        rows.append((r.stay_id, r.hour, r.event, 0, np.nan, np.nan))
        rows.append((r.stay_id, r.end_hour, r.event, 2, np.nan, np.nan))
AT = pd.DataFrame(rows, columns=['stay_id','hour','event','kind','value','rate'])
AT['minute'] = np.round(AT.hour*60).astype(int)
B = AT.groupby(['stay_id','minute']).agg(hour=('hour','first'), n=('event','size')).reset_index().sort_values(['stay_id','minute'])
SB = pd.DataFrame({'stay_id': sids_S, 'minute': np.round(hrs_S*60).astype(int), 'n_atoms': nat_S, 'row': np.arange(len(sids_S))})
MB = SB.merge(B, on=['stay_id','minute'], how='left')
print('bundles states', len(SB), 'mine', len(B), 'matched', MB.n.notna().sum(), 'n mismatch', (MB.n != MB.n_atoms).sum())
AT = AT.merge(SB[['stay_id','minute','row']], on=['stay_id','minute'], how='left')
print('atoms without row', AT.row.isna().sum())
# previous row / dt per state row
SB['prev'] = SB.groupby('stay_id').row.shift(1)
SB['dt'] = (SB.groupby('stay_id').minute.diff())  # minutes, approx
hr_ = pd.Series(hrs_S); SB['dts'] = (hr_ - hr_.groupby(SB.stay_id).shift(1)).values*3600
codes_names = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
meas_rows = set(AT[AT.event.isin(codes_names)].row) ; act_rows = set(AT[AT.event.isin(['insulin','norepinephrine'])].row)
vis_rows = set(AT[AT.event.isin(['hospital_admission','icu_admission'])].row)
only_meas = sorted([r for r in meas_rows if r not in act_rows and r not in vis_rows and not np.isnan(SB.prev[r])])
print('meas-only bundles', len(only_meas))
def ctx_rows(rws):
    rws = torch.tensor(rws); pv = torch.tensor(SB.prev.values[rws.numpy()].astype(np.int64)); dts = torch.tensor(SB.dts.values[rws.numpy()], dtype=torch.float32)
    hd = HS[pv] * torch.exp(-dts[:, None, None] * torch.exp(P['core.flow.a']))
    return OUT[pv] + ln(hd.mean(1), 'core.out_norm') + mlp((torch.log(dts)/10)[:, None], 'gap', F.gelu)
def atom_vec_meas(sub, unit_map):
    c = torch.tensor(sub.event.map(cid).values)
    v = sub.value.values; zvv = np.zeros(len(sub)); vbb = np.zeros(len(sub), np.int64)
    for ev in codes_names:
        m = (sub.event == ev).values
        if m.any(): zvv[m] = zsc(ev, 'value', v[m]); vbb[m] = fbin(ev, 'value', v[m])
    numf = torch.zeros(len(sub), 6); numf[:, 0] = torch.tensor(zvv, dtype=torch.float32); numf[:, 1] = 1
    u = torch.tensor(sub.event.map(unit_map).values)
    return (P['embed.code.weight'][c] + P['embed.kind.weight'][0] + P['embed.ekind.weight'][1] + P['embed.kq.weight'][0]
            + P['embed.categorical.unit.weight'][u] + P['embed.bins.value.weight'][torch.tensor(vbb)] + lin(numf, 'embed.numeric'))
def deepsets(av, grp, ngrp, pre):
    h1 = mlp(av, pre + '.mlp1', F.gelu)
    s = torch.zeros(ngrp, 192).index_add_(0, grp, h1)
    n = torch.zeros(ngrp).index_add_(0, grp, torch.ones(len(grp)))
    s = s / torch.sqrt(n)[:, None] + lin(torch.log1p(n)[:, None], pre + '.size')
    return ln(mlp(s, pre + '.mlp2', F.gelu), pre + '.norm')
ctxm = ctx_rows(only_meas)
subA = AT[AT.row.isin(only_meas)].copy(); rmap = {r: i for i, r in enumerate(only_meas)}
grp = torch.tensor(subA.row.map(rmap).values)
zt = ZP[torch.tensor(only_meas)]
res = []
for ug, um in itertools.permutations([2, 3, 4, 5], 2):
    unit_map = {'glucose_finger': ug, 'glucose_lab': ug, 'map_arterial': um, 'map_cuff': um}
    e_ = deepsets(atom_vec_meas(subA, unit_map), grp, len(only_meas), 'encode')
    for uzero in [True]:
        u_ = torch.zeros_like(e_)
        zp_ = mlp(torch.cat([ctxm, e_, u_], 1), 'z_post', F.gelu)[:, :32]
        res.append(((zp_ - zt).abs().mean().item(), ug, um))
res.sort(); print(res[:5], 'zt scale', zt.abs().mean().item())
```