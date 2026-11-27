Within each glucose band, the insulin rate is statistically independent of everything observed (R² ≈ 0.0001 against the previous rate, previous glucose, hour, NE, MAP and lab freshness). It looks like a band-specific base rate (1, 2, 3, ...) times random noise, rounded to 0.5. The 250-300 band has one more internal step. All 7,659 insulin starts without a fresh finger reading had a fresh lab glucose ≥ 150, so the policy reads the latest glucose of either kind. This is very good news: dosing depends only on recorded readings plus independent noise. There is no hidden confounding, and the within-band rate noise is a randomised experiment I can use to estimate the true insulin effect.

Next I'll analyse the NE policy the same way:
- which MAP it uses (arterial versus cuff);
- whether it titrates from the previous rate;
- how much residual randomness there is.

If NE also has independent noise given the recorded history, both effects are identified from the records.

```python
open('/work/notes.md','a').write("""
## Insulin policy (runs 9-12)
- Deterministic sliding scale on latest glucose (fresh gf, else fresh gl): none if <150; band base rates ~1 [150,180), ~2 [180,220), ~3 [220,250), ~3.7 [250,300) (has inner step), ~6.5 >=300.
- Within band, rate noise (0.5 steps) is independent of all observed covariates (R2~1e-4) => randomized experiment. No hidden confounding apparent.
- gf measured every hour while last glucose >150.
""")
x['map_now'] = x.ma.fillna(x.mc)
w = x[x.map_now.notna()].copy()
w['mapi'] = w.map_now.round()
t1 = w.groupby(['mapi']).agg(p=('nor', lambda s: s.notna().mean()), rate=('nor','mean'), n=('nor','size'))
print(t1.loc[50:85].T.round(3).to_string())
# conditional on previous NE status
for prevon in [False, True]:
    ww = w[(w.nor_prev>0)==prevon]
    t2 = ww.groupby(pd.cut(ww.map_now,[0,50,55,60,62,64,65,66,68,70,72,75,80,90,200]), observed=True).agg(p=('nor', lambda s: s.notna().mean()), rate=('nor','mean'), n=('nor','size'))
    print('prev on' if prevon else 'prev off'); print(t2.T.round(3).to_string())
```