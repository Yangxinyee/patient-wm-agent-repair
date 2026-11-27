Removing the four units costs essentially nothing in NLL (-0.0001 per reading), even though it moves predictions by about 0.17. So repurposing is not the problem. The cost comes from the corrections: the NE pair adds a noisy readout on every arterial row, about +0.022 per held-out arterial row, and the insulin pair costs about +0.003 per glucose_finger row.

My offline per-code costs still disagree with the tool's: glucose_finger 0.003 against 0.0062, map_arterial 0.022 against 0.0053. That could be stay mix or the tool's 4 draws against my 8.

To decide whether I can trust offline estimates for design, I'll compare offline and tool dNLL per event on the 38 tuning stays that are in both sel1 (my held-out head inputs) and fwd_11 (the tool's factual run). If they agree on the same stays, the offline evaluator is sound and the earlier gap is just stay mix.

```python
ov = sorted(set(RR.stay[src=='1'].unique()) & set(ff.stay_id.unique()))
print('overlap stays', len(ov))
mo = (src=='1') & np.isin(RR.stay.values, ov)
off_df = pd.DataFrame({'stay': RR.stay.values[mo], 'hour': RR.hour.values[mo], 'code': cs[mo], 'dn_off': dn[mo]})
ev2code = {'map_cuff':5, 'glucose_finger':6, 'map_arterial':7, 'glucose_lab':10}
tool = ff[ff.stay_id.isin(ov)].copy(); tool['code'] = tool.event.map(ev2code)
for c in [6,10,7,5]:
    a = off_df[off_df.code==c].dn_off; b = tool[tool.code==c].dnll
    print(code2ev[c], 'n off %d tool %d | dNLL/row offline %.4f tool %.4f' % (len(a), len(b), a.mean(), b.mean()))
print('overall offline %.4f tool %.4f' % (off_df.dn_off.mean(), tool.dnll.mean()))
```