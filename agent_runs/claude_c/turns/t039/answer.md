Two findings about the z dose code.
1. **Not a deviation code.** At insulin bundles, z moves along a single direction that increases with the dose (0.82 per U/h) and also with the tier base (0.60 per tier, the same direction; cosine exactly −1 with minus-base). So z encodes something like "insulin amount plus glucose level", not D − base.
2. **Not drug-specific.** The same direction varies as much at reading bundles (sd 1.65) as at drug bundles. It also tracks NE dose (r = 0.47 at NE-only bundles against 0.64 for insulin). Amplifying it through z would therefore disturb every bundle.

The cleanest drug-only input is the linear dose feature: embed.numeric columns z_dose and z_rate, which only drug atoms use. For both drugs the two columns carry identical values (insulin z = rate − 1.5; NE z = 1.695·rate − 0.78). Because the feature is centred on the median rate, amplifying it mostly shifts factual predictions by ±0.5 U/h-equivalents for tiers 1–2. My rough estimate is that the double-counting cost is near the NLL budget but may fit, since the correct noise response also gains some fit.

Next I will check that the edit functions exist and register the column-scaling edit (×5 on columns 2 and 4). Then I will test it with a dose_edit "both" run. If the insulin ×2 readout moves strongly from −1.85 towards −13 and NE moves the right way, this knob is promising; if not, I move to encoder or z_post knobs.

```python
print([n for n in ['scale','rank1','ops','undo','clear_ops'] if n in dir()])
import inspect
for n in ['scale','rank1','ops']:
    try: print(n, inspect.signature(globals()[n]))
    except Exception as e: print(n, e)
clear_ops()
e2 = np.zeros(6); e2[2]=1.0
e4 = np.zeros(6); e4[4]=1.0
rank1('embed.numeric.weight', e2, 4.0, side='in')
rank1('embed.numeric.weight', e4, 4.0, side='in')
print(ops())
open('/work/notes.md','a').write("""
## Model diagnostics (passes 1-3)
- Original dose_edit on tuning[0:100]: ins x2 -1.85, ins x1/2 +0.78, ne x2 -0.50, ne x1/2 -0.60. My estimated truth on same stay-hours:
  -13.07, +6.54, +4.75, -2.38 (population ~ -13.6, +6.8, +4.6, -2.3). Readout = pooled mean over reached stay-hours (rows only at
  recorded reading hours 13-24 of target code; reached = drug dose at h>=12 started before reading).
- Model insulin response has right time-shape but ~1/7 magnitude. NE: negative for both x2 and x1/2.
- Factual original tuning[0:100]: value NLL 1.63461 (n=4751). Model's teacher-forced slope of next glucose pred on within-tier dose:
  +0.17+-0.26 (truth -2). NE slope 0.03+-0.8 (truth ~3). => model hasn't learned causal effects at all.
- States: dose explains 14% of z posterior var at insulin bundles, ~0.3% of dOut/dH. z dose-direction also varies at reading bundles.
- Concern: amplifying full-dose pathway double counts systematic (tier-driven) dose effect absorbed in reading dynamics.
""")
```