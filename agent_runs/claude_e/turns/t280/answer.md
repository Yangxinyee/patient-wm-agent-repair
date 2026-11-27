I forgot to filter on the `reached` flag, so that table averaged over all stay-hours, including ones where the drug had not run. The direction is still clear, though.

On this batch the NE pair at gain 2 moves NE x1/2 from -0.27 to -0.54 and NE x2 from -0.05 to +0.13 on the unfiltered average. Both moves are towards the truth. Raising insulin from -2.6 to -2.7 moves the insulin readouts by about 4% (x2 -8.37 → -8.67).

Next I will redo the comparison on reached rows only, matching the tool's definition. For each edit I will report:
- the per-stay spread, as a noise estimate, since only 22 stays reach NE here;
- the edited-to-truth ratio, so I can see where each readout stands on this batch.

```python
rows = []
for e, g in d27e[d27e.reached].groupby('edit'):
    mult = 2.0 if 'x2' in e else 0.5
    f = truth_ins if e.startswith('insulin') else truth_ne
    tr = np.array([f(s, h, mult) for s, h in zip(g['stay_id'], g['hour'])], dtype=float)
    g22 = d22e[(d22e.edit==e)&d22e.reached]; go = d27o[(d27o.edit==e)&d27o.reached]
    ps = g.groupby('stay_id')['shift'].mean(); se = ps.std()/np.sqrt(len(ps))
    m = g['shift'].mean()
    rows.append((e, len(g), g.stay_id.nunique(), round(m,3), round(se,3), round(g22['shift'].mean(),3), round(go['shift'].mean(),3), round(np.nanmean(tr),3), round(m/np.nanmean(tr),3)))
print(pd.DataFrame(rows, columns=['edit','n','stays','g3(-2.7,2)','se_stay','ins_only(-2.6)','orig','truth','ratio']).to_string())
```