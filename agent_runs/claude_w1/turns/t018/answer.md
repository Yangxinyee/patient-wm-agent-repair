The insulin effect is linear in dose: the squared-shock term is about 0, with t < 1.5 at every horizon tested. Its strength grows mildly with glucose level, from about −1.83 mg/dL per u/h at G≈164 to about −2.43 at G≈272 (one hour later). The time profile is similar across levels: peak at 2 hours, faded by about 6 hours. So a mostly constant per-unit effect with a mild dependence on level. Doubling doses should then give about twice the natural-noise response, and halving about −0.5 times it.

Now norepinephrine and MAP. The norepinephrine rule titrates from the previous rate, so a random dose innovation carries forward into later rates. I'll therefore need both its effect on MAP and its persistence in the dose, then deconvolve.

First I'll pin down the details of the norepinephrine rule:
- how many rates sit at the 1.0 cap;
- the starting rate;
- when infusions stop;
- whether log(rate/previous rate) depends only on the MAP band, plus random noise unrelated to history.

```python
Nall = P[(P.ne_on==1)].copy()
Nall['m'] = Nall.map_arterial.fillna(Nall.map_cuff)
Nall['last_m'] = Nall.last_map_arterial.where(Nall.last_map_arterial.notna(), Nall.last_map_cuff)
print("frac at cap 1.0:", (Nall.norepinephrine>=1.0).mean())
# start rate
st = Nall[Nall.ne_prev_on==0]
print("start rate by MAP:", st.groupby(pd.cut(st.m,[0,40,50,55,60,65,70,200])).norepinephrine.agg(['mean','std','size']).to_string())
print(st.norepinephrine.describe())
# continuing: log ratio by fine MAP; check sharpness at 65 and 75
cont = Nall[(Nall.ne_prev_on==1)&Nall.m.notna()].copy()
cont['lr'] = np.log(cont.norepinephrine/cont.ne_prev)
cont = cont[cont.ne_prev<0.95]  # avoid cap
s = cont.groupby(cont.m.round()).lr.agg(['mean','std','size'])
print(s.loc[55:85].T.round(3).to_string())
# stopping rule
P['ne_next_on'] = P.groupby('stay_id').ne_on.shift(-1)
stp = P[(P.ne_on==1)&(P.h<24)].copy()
stp['m'] = stp.map_arterial.fillna(stp.map_cuff)
print(stp.groupby([pd.cut(stp.groupby('stay_id').map_arterial.shift(-1).fillna(stp.groupby('stay_id').map_cuff.shift(-1)), [0,65,70,75,80,85,90,200]), pd.cut(stp.norepinephrine,[0,0.05,0.1,0.2,0.5,1.0])]).ne_next_on.mean().unstack().round(3).to_string())
```