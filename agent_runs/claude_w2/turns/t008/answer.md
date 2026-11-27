The grid works. Drug records run at hours 0 to 24 and readings at -1 to 24. Insulin use grows over the stay, from about 9,500 to 12,700 stays per hour, while NE stays flat at around 9,500. Readings are irregular:
- glucose_finger in about 55% of stay-hours;
- glucose_lab in about 20%;
- map_arterial in about 45% (probably the stays with an arterial line);
- map_cuff in about 61%.

Next I'll characterise the dosing policy, because it decides which causal method is valid. I'll regress whether insulin is given, and its rate, on the latest readings and the previous rate, and check how much of the dose is predictable from the recorded history. If dosing is a deterministic function of recorded readings plus independent noise, adjusting for history identifies the effect. If it is not, there may be hidden confounding and I'll need a natural experiment.

```python
gg = g.copy()
gg = gg.sort_values(['stay_id','h'])
for c in ['gf','gl','ma','mc']:
    gg[c+'_ff'] = gg.groupby('stay_id')[c].ffill()
gg['ins0'] = gg.ins.fillna(0); gg['ne0'] = gg.ne.fillna(0)
gg['ins_prev'] = gg.groupby('stay_id').ins0.shift(1)
gg['ne_prev'] = gg.groupby('stay_id').ne0.shift(1)
x = gg[gg.h>=1].copy()
# insulin given vs current gf (forward-filled)
x['gfb'] = pd.cut(x.gf_ff, [0,100,120,140,160,180,200,220,250,300,400,1000])
print(x.groupby('gfb').agg(p_ins=('ins', lambda s: s.notna().mean()), rate=('ins','mean'), n=('ins','size')).to_string())
print(pd.crosstab(x.ins_prev, x.ins0.round(1)).iloc[:12,:12])
x['mab'] = pd.cut(x.ma_ff.fillna(x.mc_ff), [0,50,55,60,65,70,75,80,90,200])
print(x.groupby('mab').agg(p_ne=('ne', lambda s: s.notna().mean()), rate=('ne','mean'), n=('ne','size')).to_string())
```