Norepinephrine follows a titration policy, unlike insulin's tier table.
- **Start:** NE starts when MAP < 65 (always), about 56% of the time at 64–65, and rarely above. Lower MAP gives a higher starting rate.
- **Continue:** once running it continues about 99% of the time while MAP < 70, about 98% at 70–75, and about 81% at 75–90, so weaning is slow.
- **Titration:** the new rate is the previous rate times a ratio with mean 1.08 and sd 0.37.
- **No reading needed:** NE also runs in hours with no MAP reading (13,019).

The exogenous variation must therefore be the noise in the titration step around a rule that depends on the previous rate and the current MAP.

Next I will pin down that rule:
1. Which MAP reading drives decisions when both arterial and cuff exist?
2. How does log(rate/previous rate) depend on MAP? I expect up-titration below 65, holding in the middle, weaning above about 75.
3. What happens when no MAP reading exists?

The aim is to isolate a residual that is unrelated to pre-treatment variables, as with insulin.

```python
q = P[P.index.get_level_values('h')>=0].copy()
q['NEp'] = P.groupby(level=0).NE.shift(1).loc[q.index]
# start decision: no previous NE
s0 = q[q.NEp.isna()]
for src in ['map_a','map_c']:
    other = 'map_c' if src=='map_a' else 'map_a'
    x = s0[s0[src].notna() & s0[other].notna()]
    x = x[((x[src]<65) != (x[other]<65))]
    print(src, "<65 disagreement cases", len(x), " p_start when src<65:", x[x[src]<65].NE.notna().mean(), " when src>=65:", x[x[src]>=65].NE.notna().mean())
# continuing: log ratio vs MAP
c = q[q.NEp.notna() & q.NE.notna()].copy()
c['lr'] = np.log(c.NE/c.NEp)
c['m'] = c.map_a.fillna(c.map_c)
c['src'] = np.where(c.map_a.notna(),'a',np.where(c.map_c.notna(),'c','none'))
c['mb'] = pd.cut(c.m,[0,45,50,55,60,65,70,75,80,85,90,200])
print(c.groupby(['src','mb'],dropna=False).lr.agg(['size','mean','std']).round(3).to_string())
print("at cap: frac NEp==1.0", (c.NEp==1.0).mean())
```