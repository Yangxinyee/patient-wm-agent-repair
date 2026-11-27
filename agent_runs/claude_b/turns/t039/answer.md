The edited model gives exactly 0.0 for all four dose readouts on the same 100 stays. The original reproduces its earlier +0.141, −0.084, +0.006 and +0.015 exactly, since rollouts are deterministic. Combined with the unchanged factual NLL, the blinding edit works mechanically: dose size no longer reaches the predictions at all.

The remaining question is whether 0 is truly right. My one-hour insulin estimate is +0.034 ± 0.043 per u/h. If an effect that small persisted and summed over the many doubled doses from hour 12 to 24, the true insulin ×2 readout could still be several tenths of a mg/dL. I have plenty of budget, so I will check two things:

1. **Cross-effects** (norepinephrine on glucose, insulin on MAP). These tell me whether this simulator has any drug effects at all. Effects there would show that drugs can act in this ICU and that the nulls on the targets are specific. Nulls everywhere would suggest the drugs are simply inert.
2. **Cumulative readout effect** (next run). I will estimate it directly with a stay-level bootstrap.

```python
# (a1) NE -> glucose: use titration surprise (joint FE on MAP_t x NE_{t-1}); outcome finger glucose change
B['G_t'] = P.glucose_finger.reindex(B.index).values
gfP = P.glucose_finger.groupby(level=0)
for k in [1,3,6]:
    B[f'Gf{k}'] = gfP.shift(-k).reindex(B.index).values
    y = (B[f'Gf{k}'] - B.G_t).values
    m = ~np.isnan(y)
    b,se,n = fe_slope(y[m], B['ne'].values[m], B.grp.values[m], B.stay.values[m])
    print(f"NE_t -> glucose change k={k}: {b:+.3f} +- {se:.3f} n={n}")
# (a2) insulin -> MAP (arterial): insulin surprise via FE on finger reading; outcome arterial MAP change
D['M_t'] = P.map_arterial.reindex(D.index).values
for k in [1,3,6]:
    D[f'Maf{k}'] = maP.shift(-k).reindex(D.index).values
    y = (D[f'Maf{k}'] - D.M_t).values; m = ~np.isnan(y)
    b,se,n = fe_slope(y[m], D.ins.values[m], D.glucose_finger.round().astype(int).values[m], D.stay.values[m])
    print(f"insulin_t -> MAP change k={k}: {b:+.3f} +- {se:.3f} n={n}")
# (a3) does MAP level affect glucose dynamics or vice versa (are the two systems coupled at all)?
DD = D[D.M_t.notna() & D.dR1.notna()]
b,se,n = fe_slope(DD.dR1.values, DD.M_t.values, DD.glucose_finger.round().astype(int).values, DD.stay.values)
print(f"glucose change ~ current MAP: {b:+.4f} +- {se:.4f}")
```