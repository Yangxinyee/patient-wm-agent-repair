Solved: glucose_finger atoms carry no unit at all (row 0), and with that the single-atom error is 5e-7, i.e. exact. The measurement conventions are now fully known:
- z_value in numeric column 0 and has_value in column 1;
- code, kind 0, event kind 1 and kq 0 rows added, plus the value-bin row;
- unit row 2 for both MAP codes, row 5 for glucose_lab and none for glucose_finger;
- u = the empty-action encoding.

Next I will check every measurement-only bundle, including multi-atom ones, to confirm the 1/√n and size(log(1 + n)) aggregation. Then I will start on action bundles, whose revealed value is u_k plus the empty-measurement encoding. For each drug atom I will search the rate-unit and unit rows, the numeric columns for dose and rate, and how END atoms are built. An exact multi-atom match will confirm the aggregation. Then one action configuration should come out exact as well.

```python
UNIT = {'glucose_finger': 0, 'glucose_lab': 5, 'map_arterial': 2, 'map_cuff': 2}
def meas_atoms(sub_):
    c_ = torch.tensor(sub_.event.map(cid).values); v_ = sub_.value.values; zv_ = np.zeros(len(sub_)); vb_ = np.zeros(len(sub_), np.int64)
    for ev in codes_names:
        m = (sub_.event == ev).values
        if m.any(): zv_[m] = zsc(ev, 'value', v_[m]); vb_[m] = fbin(ev, 'value', v_[m])
    nf = torch.zeros(len(sub_), 6); nf[:, 0] = torch.tensor(zv_, dtype=torch.float32); nf[:, 1] = 1
    return (P['embed.code.weight'][c_] + P['embed.kind.weight'][0] + P['embed.ekind.weight'][1] + P['embed.kq.weight'][0]
            + P['embed.categorical.unit.weight'][torch.tensor(sub_.event.map(UNIT).values)] + P['embed.bins.value.weight'][torch.tensor(vb_)] + lin(nf, 'embed.numeric'))
subM = AT[AT.row.isin(only_meas)].copy()
rmM = {r: i for i, r in enumerate(only_meas)}; gM = torch.tensor(subM.row.map(rmM).values)
avM = meas_atoms(subM)
for agg in ['sqrt', 'mean', 'sum']:
    h1 = mlp(avM, 'encode.mlp1', F.gelu); s = torch.zeros(len(only_meas), 192).index_add_(0, gM, h1); n = torch.zeros(len(only_meas)).index_add_(0, gM, torch.ones(len(gM)))
    s = s / (torch.sqrt(n) if agg == 'sqrt' else (n if agg == 'mean' else torch.ones_like(n)))[:, None] + lin(torch.log1p(n)[:, None], 'encode.size')
    eM = ln(mlp(s, 'encode.mlp2', F.gelu), 'encode.norm')
    print(agg, 'all meas-only bundles err', (eM[:, :32] + u_empty[:, :32] - REV[torch.tensor(only_meas)]).abs().mean().item())
# action bundles inventory
act_only = sorted([r for r in act_rows if r not in meas_rows and r not in vis_rows])
subX = AT[AT.row.isin(act_only)].copy()
print('action-only bundles', len(act_only), 'atoms', len(subX))
print(subX.groupby(['event', 'kind']).size())
print(subX.head(8).to_string())
```