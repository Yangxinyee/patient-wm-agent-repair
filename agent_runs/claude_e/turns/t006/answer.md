The hourly grid shows readings at hours −1 to 24 and infusion records at hours 0 to 24. Glucose_finger is present in 51–59% of hours, and that share rises over the stay, as does the share of hours with insulin. That hints that finger-sticks become more frequent when insulin is running, so measurement may be informative.

Now the key question is how clinicians choose doses. I'll look at:
- when insulin starts and stops,
- how its rate relates to the latest glucose readings (finger and lab) and to the previous rate,
- the same for norepinephrine against MAP.

If the dose is a near-deterministic function of recorded values, there is little hidden confounding but also little free variation. The structure of the leftover variation tells me which identification strategy is possible.

```python
def last_obs(G, key):
    # most recent value at or before hour h (forward fill along hours)
    return G[key].ffill(axis=1)
H = list(range(0,25))
rows = []
gf_ff = last_obs(Gtr,'gf'); gl_ff = last_obs(Gtr,'gl'); ma_ff = last_obs(Gtr,'ma'); mc_ff = last_obs(Gtr,'mc')
ins_prev = Gtr['ins'].shift(1, axis=1); ne_prev = Gtr['ne'].shift(1, axis=1)
for h in H:
    rows.append(pd.DataFrame({'stay':stays_tr,'h':h,'gf_now':Gtr['gf'][h].values,'gf_last':gf_ff[h].values,'gl_last':gl_ff[h].values,
        'ma_now':Gtr['ma'][h].values,'ma_last':ma_ff[h].values,'mc_last':mc_ff[h].values,'ins':Gtr['ins'][h].values,'ins_prev':ins_prev[h].values,
        'ne':Gtr['ne'][h].values,'ne_prev':ne_prev[h].values,'gf_next':Gtr['gf'][h+1].values,'ma_next':Gtr['ma'][h+1].values}))
P = pd.concat(rows, ignore_index=True)
P['on_ins'] = P.ins.notna()
b = pd.cut(P.gf_last, [0,100,120,140,160,180,200,220,250,300,500])
print(P.groupby(b, observed=True).agg(n=('ins','size'), p_ins=('on_ins','mean'), mean_rate=('ins','mean'), sd_rate=('ins','std')).to_string())
print(P.groupby(P.ins_prev.notna()).on_ins.mean())
```