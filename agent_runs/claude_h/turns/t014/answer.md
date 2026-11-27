The local projections show no detectable insulin effect at any horizon from 1 to 8 hours: every coefficient is within about ±0.1 mg/dL per u/h, with SEs of 0.04–0.08. Later doses also do not respond to the jittered dose (no feedback). Taken at face value, the jittered insulin rate has no effect on finger glucose in this ICU, so the true x2 and x1/2 effects would be about 0. I still need to rule out nonlinear forms, such as a saturating or threshold effect or an effect only at high doses. I will also check lab glucose as a cross-check.

Next I will characterise the norepinephrine policy against MAP (arterial versus cuff, thresholds, jitter) in the same way. A clear MAP scale with iid jitter would let me estimate its effect by the same local-projection design. I am also writing the findings so far to the notebook.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 3-13: data structure & insulin
- Hourly grid: readings at integer hours -1..24, drug records at h+0.0833 (run 1h). Stays end 25.17.
- Insulin policy: sliding scale on finger glucose (lab if no finger): base 0 (<150),1 (150-180),2 (180-220),3 (220-260),4 (260-300),6 (>=300),
  times iid multiplicative jitter (0.5..2, median 1; autocorr 0.005, no stay effect, uncorrelated with trend); ~3% omissions. No glucose reading -> no insulin.
  If insulin at t-1 -> finger reading at t always present (else 31%).
- Panel saved /work/panel_train.pkl, /work/panel_tune.pkl.
- KEY: with insulin>0 sample & 5mg/dL glucose-bin FE, jitter-identified effect of rate on G(t+k)-G(t), k=1..8: all ~0 (+0.001,-0.03,+0.06,+0.04,+0.08,+0.01,+0.11,-0.02; se .04-.08 mg/dL per u/h).
  Naive association is negative (-0.75) from regression to mean. => Insulin appears to have ~NO effect on finger glucose. (to verify: nonlinearity, lab)
""")
d = P[P.ma.notna() | P.mc.notna()].copy()
d['map_any'] = d.ma.combine_first(d.mc)
d['mbin'] = pd.cut(d.ma, [0,40,50,55,60,62,65,68,70,75,80,90,200])
print(d.groupby('mbin').ne.agg(['mean','std','size', lambda x:(x>0).mean()]).to_string())
dc = d[d.ma.isna()]
dc['cbin'] = pd.cut(dc.mc, [0,40,50,55,60,62,65,68,70,75,80,90,200])
print("cuff only:"); print(dc.groupby('cbin').ne.agg(['mean','std','size', lambda x:(x>0).mean()]).to_string())
print("ne when no map reading:", P.loc[P.ma.isna()&P.mc.isna(),'ne'].agg(['mean','size']).to_dict())
print("P(map_art at t| ne_prev>0)", P.loc[P.ne_prev>0,'ma'].notna().mean(), "P(any map| ne_prev>0)", (P.loc[P.ne_prev>0,'ma'].notna()|P.loc[P.ne_prev>0,'mc'].notna()).mean())
```