The regressions came out internally inconsistent, which points to a bookkeeping error rather than a finding. For finger glucose, the dose-and-policy parametrization gives +2.9 on lag-1 dose and −3.8 on lag-1 policy. The deviation-and-policy parametrization gives −1.2 on lag-1 deviation and −0.5 on lag-1 policy. If dose = policy + deviation at the same lag, these two fits must agree exactly, and they don't. Norepinephrine shows the same contradiction.

So the lag columns I built by shifting ins_base / e_ins / nexp must not line up with gg's own ins_lk / nor_lk columns. Most likely gg's "l0" refers to a different hour than I assumed (for example, the infusion running *before* reading h rather than the one started after it).

I will print gg for one insulin stay next to its raw insulin records, and check ins_l1 against the groupby-shift of ins_l0 and against ins_base + e_ins. I will not interpret the coefficients until the alignment is settled. The test is simple: if ins_l1 equals ins_l0 shifted by one hour, the shift is fine and the problem is elsewhere (e.g. ins_base's definition).

```python
sid=tr[(tr.event=='insulin')].stay_id.iloc[500]
a=gg[gg.stay_id==sid][['h','glu_now','gf','gl','ins','ins0','ins_l0','ins_l1','ins_l2','ins_base','e_ins','nor','nor0','nor_l0','nor_l1','nor_prev','e_nor','map_pol']]
print(a.head(16).round(2).to_string())
r=tr[(tr.stay_id==sid)&(tr.event.isin(['insulin','glucose_finger']))][['hour','event','value','rate','end_hour']]
print(r.head(24).to_string())
G=gg.sort_values(['stay_id','h']); gb=G.groupby('stay_id')
for c in ['ins_l0','ins','ins0']:
    s=gb[c].shift(1); ok=s.notna()&G.ins_l1.notna(); print(c,'shift1 vs ins_l1 maxdiff',np.abs(s[ok]-G.ins_l1[ok]).max(), 'n',ok.sum())
print('ins_base NaN frac',G.ins_base.isna().mean(),'e_ins NaN frac',G.e_ins.isna().mean())
```