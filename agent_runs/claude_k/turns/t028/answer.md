The tuning set has 217 stays with insulin from hour 12, 153 with norepinephrine from hour 12, and 87 with both. The norepinephrine readout is on map_arterial, which needs an arterial line, and 96 of the norepinephrine stays have one.

Next I'll update the notebook, then pick 100 tuning stays: the 87 with both drugs, topped up with norepinephrine-from-12 stays that have an arterial line. I'll run the evaluation's dose_edit rollout on the original model. That shows how the model currently answers all four dose questions and which stay-hours the readout uses. I can then compute my records-based truth on exactly those stay-hours. My hypothesis is that the original model gets the insulin effect wrong (too small or wrong-signed) and likely misstates norepinephrine too. If its shifts already match about -12 mg/dL scale insulin effects and +4 mmHg norepinephrine effects, little editing is needed.

```python
open('/work/notes.md','a').write("""
## Effect estimates (runs 20-26): cross-fitted double-ML with GBM on history feats, deconvolved by first stage
- INSULIN -> glucose impulse h_k (k=1..12, per unit/h for 1h): -1.99,-2.39,-2.11,-1.72,-1.22,-0.95,-0.69,-0.55,-0.35,-0.15,-0.12,-0.05
  ~ -1.99*k*0.6^(k-1); sum ~ -12.3 per sustained unit/h. Homogeneous across subgroups (ne on/off, art, time, glucose band). Linear in dose.
- NOREPI -> MAP impulse h_k: 3.35,2.60,1.75,1.03,0.42,0.27, ~0 after (sum ~9.6 per sustained 1 mcg/kg/min).
  Heterogeneity: lag-1 effect lower (~2.5-3.4) when MAP<60; ~3.6-4 otherwise; no clear rate saturation in other strata.
- Cross effects ~0 (insulin->MAP 0; norepi->glucose +0.6+-0.25 lag1 only).
- Tuning: 217 stays insulin>=12, 153 norepi>=12, 87 both, 96 norepi>=12 w/ arterial line.
""")
both_ids = [ids_t[i] for i in np.where(has_ins12 & has_ne12)[0]]
extra = [ids_t[i] for i in np.where(has_ne12 & art_t & ~has_ins12)[0]][:100-len(both_ids)]
sel1 = both_ids + extra
print(len(sel1)); print(json.dumps(sel1))
json.dump(sel1, open('/work/sel1.json','w'))
```