The start-threshold comparison does not support a vasopressor effect. One to six hours after an arterial MAP just below 65 (norepinephrine starts), MAP is slightly lower, not higher, than just above 65: −0.1 to −0.8 mmHg arterial. The rate jump at the threshold is small (about 0.06 mcg/kg/min), so this design has little power and the scaled ratios are meaningless.

The innovation analysis gave a small positive association that decays over 3–8 hours on arterial MAP but persists on cuff MAP. Decay with lag, while the rate shock itself persists, is the signature of confounding by a transient state, not of a drug effect. My crude band-mean policy model is the prime suspect.

So the norepinephrine question is still open, with the evidence pointing to a small or zero effect. To build a correct policy model, I will now look directly at a few norepinephrine stays, both with and without arterial lines:
- the rate sequence next to the MAP readings;
- how steps, rounding, caps and stops behave.

I will also check the threshold behaviour for cuff-only stays.

```python
pd.set_option('display.width', 250)
ne_stays = r.stay_id.unique()
art_st = st.map_arterial
for sid in [ne_stays[3], ne_stays[10]]:
    dd = PN.loc[sid][['map_arterial','map_cuff','norepinephrine']].T
    print(sid, "arterial stay:", art_st.loc[sid])
    print(dd.round(2).to_string())
# cuff-only stays: titration vs cuff
co = r[~r.stay_id.map(art_st)]
print("cuff-only stays NE records:", len(co))
print(co.groupby(pd.cut(co.map_cuff,[0,50,55,60,62,64,65,66,68,70,72,75,77,80,90,200])).lratio.agg(['mean','std','count']).round(3).to_string())
```