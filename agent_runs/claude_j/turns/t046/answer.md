The vocab settles two tokenisation questions:
- **z-scores are linear.** z = (x − median)/iqr, clipped to ±10. For insulin, z_dose = z_rate = dose − 1.5, so the numeric embedding is linear in dose. For norepinephrine, z_rate = (rate − 0.46)/0.59.
- **Fine bins don't match the sliding scale.** Bins count the edges below x. The glucose_finger edges do not line up with the thresholds 150/180/220/300; only 260 coincides.
- **Norepinephrine x2 is far off-support.** Its rate bins 13–16 (rate > 1.0) never occur in training, and a doubled rate gives z up to 2.6 versus a factual maximum of 0.92.
- **Predicted means are bounded.** They come from bin centres, so they cannot go above 272 or below 73.

My plan is a CPU replica of the model, checked against its real internal states, so I can engineer and test a compensated dose pathway offline. Reference states for the same 100 tuning stays will validate the replica bundle by bundle. They also let me check how well the model's state already encodes recent insulin, the expected insulin, and their difference.

First, I'll record what I've established in the notebook so a restart can't lose it.

```python
add = """
## Session 2 findings (runs 29-42, fwd 2)
- fwd_2: factual original, tuning sel1 (100 stays): value NLL 1.51162 (n=5214); GF 1.5057, GL 1.31, MA 1.3501, MC 1.7898.
- TRUTH on sel1 (kernel KI, KN): ins x2 -13.95, x1/2 +6.98, NE x2 +4.28, x1/2 -2.14. Model (orig): -2.75, +1.38, +0.69, -0.58.
  Model/true ratio ~0.26 at lag1 and ~0.16-0.23 later (insulin); NE lag1 ratio 0.11.
- Lag-1 effect re-checked on train, ONLY hours where insulin given (base>0): slope -1.93 (se .05) per unit/h
  (bands -1.86/-2.01/-1.95/-2.36). [Including GF>150-but-no-insulin rows biases to -1.0: ~8.8k such rows.]
- Model's factual within-band dose-deviation slope at lag1 (given-only, tuning 100 stays): -0.57 (se .31) vs truth -1.93.
  => model under-uses dose ~27%; its reading dynamics absorb expected insulin. Picture: pred_old ~ N(g) - 1.46*c(g) - 0.54*d,
  truth N(g) - 2d. Needed edit term: -1.46*sum_j a^(t-j)*(dose_j - c(GF_j)) (zero-mean on factual data; c=E[dose|GF]).
- vocab: z=(x-median)/iqr clipped +-10 (LINEAR). insulin dose/rate median 1.5 iqr 1; NE rate median .46 iqr .59, dose 2.208/2.832.
  fine bin = 1 + #edges below x. GF edges [78,87,96,109,128,143,154,164,173,183,195,214,230,243,260] (not aligned w/ 150/180/220).
  GF centres [73..272] -> predicted mean bounded 73..272. NE rate edges cap at 1.0 -> x2 NE rates (>1) unseen bins.
- Plan: build CPU replica of the model (validate vs 'states' request), then engineer a compensated dose channel.
"""
open('/work/notes.md','a').write(add); print('ok')
```