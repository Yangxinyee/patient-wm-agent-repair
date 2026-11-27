The natural experiment gives a clean first estimate. Given the base and current glucose, each extra 1 u/h of insulin at hour t lowers finger glucose by about 2.0 mg/dL at t+1. The total effect is about −2.2 to −2.4 at t+2 for bases 2–3 and −1.8 at t+3. The raw means agree: within base 2, the next-hour change falls steadily from +1.1 at 0.5 u/h to −8 at 5 u/h. Glucose is very persistent (the coefficient on current glucose is about 0.95–1.0), so the effect carries on and adds up hour after hour under a sustained dose change.

Doubling a typical 2 u/h for 12 hours could therefore push glucose down by tens of mg/dL by hour 24. The exact numbers need a proper dynamic (structural) model that keeps later doses fixed, as the counterfactual does.

Next I will analyse the norepinephrine policy in the same way:
- which MAP drives it (arterial or cuff);
- how the rate depends on MAP and the previous rate;
- whether it has a random component I can use.

```python
c['Mdec_a'] = c.ma; c['Mdec_c']=c.mc
gg = c.groupby(level='stay_id')
c['nor_prev'] = gg.nor0.shift(1)
# NE presence by (ma available?, mc available?)
for nm_,cond in [('ma only',c.ma.notna()&c.mc.isna()),('mc only',c.ma.isna()&c.mc.notna()),('both',c.ma.notna()&c.mc.notna()),('none',c.ma.isna()&c.mc.isna())]:
    print(nm_, cond.sum(), 'p_ne', c.nor[cond].notna().mean().round(4))
# NE start (prev 0) as function of map_arterial (ma only hours) and of cuff (mc only)
st = c[(c.nor_prev==0)]
for col in ['ma','mc']:
    s = st[st[col].notna()]
    print(col, s.groupby(pd.cut(s[col],[0,50,55,60,62,64,65,66,68,70,75,80,200]),observed=True).agg(n=('nor0','size'),p=('nor',lambda x:x.notna().mean()),r=('nor','mean')).round(3).to_string())
# continuing: prev>0
co = c[(c.nor_prev>0)&c.ma.notna()]
print(co.groupby(pd.cut(co.ma,[0,50,55,60,65,70,75,80,85,90,200]),observed=True).agg(n=('nor0','size'),p=('nor',lambda x:x.notna().mean()),r=('nor','mean'),rprev=('nor_prev','mean')).round(3).to_string())
```