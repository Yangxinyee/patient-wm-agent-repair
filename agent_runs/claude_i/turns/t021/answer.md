The feedback-corrected kernel is clean. Lab-based (selection-free) and finger-based estimates agree up to lag 8. One hour of insulin at 1 u/h lowers glucose by:

| lag (h) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10–12 |
|---|---|---|---|---|---|---|---|---|---|---|
| effect (mg/dL) | −2.0 | −2.35 (peak) | −2.1 | −1.6 | −1.3 | −0.9 | −0.6 | −0.5 | ≈ −0.3 to −0.5 | ≈ 0 |

So the effect is transient, not a permanent integrator. Feedback through later doses is small (D ≈ −0.05). Summed over 12 h, one u/h-hour is worth about −12 mg/dL·h. Continuous doubling should therefore build up a cumulative drop. That is likely larger than the model's −2.0 average, so I'll compute the implied truth on the same 79 stays later.

First I'll establish the norepinephrine policy, since its random component is what will identify the norepinephrine effect. Rates run from 0.01 to 1.0 in 0.01 steps, with a cap at 1.0, and the drug looks titrated. I'll look at:
- what drives starting, continuing and the size of rate changes (MAP at h, previous rate);
- whether the changes have an iid random part.

```python
N = P.copy()
gN = N.groupby(level='stay_id')
N['mdec_a'] = N.ma; N['mdec'] = N.ma.combine_first(N.mc)   # guess: arterial preferred
N = N[N.index.get_level_values('h')>=0]
# start decisions: ne_prev==0
st = N[N.ne_prev==0]
print("P(start) by MAP (any) at h:", st.groupby(pd.cut(st.mdec,[0,50,55,58,60,62,65,70,80,200]),observed=True)['ne'].apply(lambda x: x.notna().mean()).round(3).to_dict())
print("P(start) no MAP reading at h:", st[st.mdec.isna()]['ne'].notna().mean().round(3))
# when both ma & mc present, which drives start?
b2 = st[st.ma.notna()&st.mc.notna()]
for nm, mm in [('ma',b2.ma),('mc',b2.mc)]:
    print(nm, "agreement start iff <60:", ((mm<60)==b2['ne'].notna()).mean().round(4), len(b2))
print("start rates:", st['ne'].describe().round(3).to_dict())
# continuing: ne_prev>0
co = N[N.ne_prev>0].copy()
co['stop'] = co['ne'].isna()
print("P(stop) by MAP:", co.groupby(pd.cut(co.mdec,[0,50,60,65,70,75,80,85,90,200]),observed=True).stop.mean().round(3).to_dict())
print("P(stop) no MAP at h:", co[co.mdec.isna()].stop.mean().round(3))
co2 = co[co['ne'].notna()]
co2['ratio'] = co2['ne']/co2.ne_prev
print("rate ratio by MAP:", co2.groupby(pd.cut(co2.mdec,[0,50,55,60,65,70,75,80,90,200]),observed=True).ratio.describe()[['count','mean','std','25%','50%','75%']].round(3).to_string())
print("rate ratio no MAP:", co2[co2.mdec.isna()].ratio.describe().round(3).to_dict())
```