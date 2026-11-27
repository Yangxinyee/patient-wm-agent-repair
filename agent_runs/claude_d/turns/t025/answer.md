Under the linear, additive dose response, the true insulin readouts on these 100 tuning stays come out at:

| edit | implied true shift | model's shift |
|---|---|---|
| insulin x2 | −14.5 mg/dL | −1.73 |
| insulin x1/2 | +7.3 mg/dL | +0.91 |

The true shift builds from −3.6 at hour 13 to a plateau of about −17 by hour 19. The model's shift creeps from −0.6 to −2.5. So the model has the right sign but only about 1/8 of the size, and the edit will need to greatly strengthen and reshape the model's response to insulin rate.

Before more insulin checks, I will set up the norepinephrine side the same way. That needs E[nor_s | recorded history] so I can form the titration-noise instrument. First I will pin down which MAP the clinician acts on (arterial if present, else cuff?), using the start and titration rules split by which readings exist that hour. I will also log the insulin findings in the notebook.

```python
open('/work/notes.md','a').write("""
## Findings so far (runs 4-23)
- Insulin policy: base = step(G_dec) with G_dec = gf at hour t if measured else gl at t; steps 150/180/220/260/300 -> 1,2,3,4,6 u/h; rate = round0.5(base*m), m~1.04+-0.33 iid per hour, unrelated to observables (R2~0). ~3-5% omissions (mostly first triggers after gap).
- gf measurement: always (100%) at t if insulin at t-1, else 30% random. -> IPW weights 1 / 1/0.3.
- NE policy: start if map_arterial<65 (rate higher for lower MAP); titration rate_t = rate_{t-1}*f(MAP)*noise (lognormal sd~0.29), f~1.15 (<65), 1.0 (65-75), 0.85 (>=75); cap 1.0; stops when rate <~0.08.
- Cross effects negligible (ins->MAP 0.01/u/h, NE->glucose 0.47 per mcg/kg/min).
- G-estimation (IV = multiplier residual, IPW) insulin impulse response beta_k (mg/dL per u/h at lag k): 
  [-1.99,-2.39,-2.11,-1.71,-1.21,-0.92,-0.64,-0.47,-0.20,-0.13,-0.08,-0.01] (se .05-.16). Saved /work/beta_ins.json
- Fwd 1 (dose_edit original, tuning sel1 100 stays, /work/sel1.json): model ins_x2 -1.73, ins_half +0.91, ne_x2 -0.075, ne_half -0.48.
  Implied TRUE on same stay-hours (linear SNMM): ins_x2 -14.5, ins_half +7.26. Model underestimates insulin effect ~8x.
""")
# NE decision source
c['ma_av']=c.ma.notna(); c['mc_av']=c.mc.notna()
st = c[(c.nor_prev==0)].copy()
for nm_,cond in [('ma only', st.ma_av & ~st.mc_av), ('both', st.ma_av & st.mc_av), ('mc only', ~st.ma_av & st.mc_av), ('none', ~st.ma_av & ~st.mc_av)]:
    s=st[cond]; col = 'ma' if 'ma' in nm_ or nm_=='both' else 'mc'
    if nm_=='none': print('none: p start', s.nor.notna().mean(), len(s)); continue
    print(nm_, s.groupby(pd.cut(s[col],[0,55,60,62,63,64,65,66,67,68,70,75,200]),observed=True).nor.apply(lambda x: x.notna().mean()).round(3).to_dict())
s = st[st.mc_av & ~st.ma_av]
print('mc only, start by mc bins fine', s.groupby(pd.cut(s.mc,np.arange(55,72,1)),observed=True).nor.apply(lambda x: x.notna().mean()).round(3).to_dict())
```