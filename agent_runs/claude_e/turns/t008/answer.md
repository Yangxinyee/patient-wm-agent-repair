The dosing protocol is now clear. It is a sliding scale on the finger-stick taken that hour:
- below 150 mg/dL: no insulin;
- 150–180: mean 1.05 u/h;
- 180–220: mean 2.09;
- 220 and above: mean 3.14;
- probably a higher tier above about 300.

Within a tier the spread is proportional to the tier, which looks like base × random factor, rounded to 0.5. There are no hours with a missing finger-stick and a last reading above 150. So patients on insulin get a finger-stick every hour, and missingness is driven by recorded values.

This gives two potentially independent identification routes:
1. the dose jumps at 150, 180 and 220 (regression discontinuity);
2. the random within-tier variation.

If both agree on the drug effect, I can trust it. Next I'll look at the within-tier rate distribution, and at whether the within-tier noise depends on anything recorded: previous rate, glucose_lab, hour, stay.

```python
Pn['tier'] = pd.cut(Pn.gf_now, [0,150,180,220,300,1000], right=False, labels=['<150','150-180','180-220','220-300','300+'])
print(pd.crosstab(Pn.tier, Pn.ins.fillna(0)).to_string())
# precise threshold check around 150, 180, 220
for th in [150,180,220,300]:
    s = Pn[(Pn.gf_now>=th-3)&(Pn.gf_now<=th+2)]
    print(th, s.groupby('gf_now').agg(n=('ins','size'), p=('on_ins','mean'), r=('ins','mean')).T.round(3).to_string())
```