Each code has its own 16 bin centres:
- glucose_finger runs from about 73 to 272 mg/dL, and glucose_lab sits about 5% lower;
- map_arterial runs from about 33 to 101 mmHg, and map_cuff from about 37 to 109.

So one shared 17-class tilt vector can't be calibrated for both families. That favours keeping a separate gated channel per drug, each with its own tilt vector:
- **Glucose channel.** Tilt along glucose centres divided by 100.
- **MAP channel.** Tilt along MAP centres.

The gates must be built to survive bf16. My plan for E3:
- Each channel uses a pair of hidden units with pre-activations κ(g_family + g_presence − 1.5) + s and κ(g_family + g_presence − 1.5).
- κ is about 8, so an active gate sits near +4, inside GELU's linear range, and a closed gate sits near −4 to −12.
- The output columns are ±λ·t.
- The bf16 rounding error in the difference is then about 0.03, worth roughly 0.3 mg/dL. That is noise, not bias.
- The dose signal s is a decoder fitted to the desired tilt, which is the mean shift the records require divided by the row's predictive variance.

Before building it I need the exact calling conventions of my existing helpers for the z-sampled with_code (wc_from, aug_wc) and for scoring NLL. I will print only their parameter names, and check that wc_from at the prior-mean z reproduces WC0.

```python
for fn_ in ['wc_from','aug_wc','cvdec_aug2','ridge_gate','fit_dec','lin_gate','exp_tilt2','eval_edit','nll_of','nll_avg','head_probs','build2','regm']:
    try: print(fn_, list(inspect.signature(globals()[fn_]).parameters.keys()))
    except Exception as ex: print(fn_, 'ERR', ex)
try:
    t0_ = wc_from(HI['zmu'][:50]) if len(inspect.signature(wc_from).parameters)==1 else None
    if t0_ is not None: print('wc_from(zmu) vs WC0 maxdiff', np.abs(np.asarray(t0_) - WC0[:50]).max())
except Exception as ex: print('wc_from test ERR', ex)
```