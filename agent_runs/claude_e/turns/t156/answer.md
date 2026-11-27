The current NE rate is much easier to read from the value head's input than the deviation. On map_arterial rows the stay-grouped CV R2 is about 0.60 for the lag-1 rate and 0.61 for lag-2, against 0.17 for nedev1. That supports rebuilding the NE correction around the rate level.

To build it I need my existing edit-construction helpers (gated3, pair, fit_ro5, head_run, eval_edit and others). Their source code can't be shown, but their signatures can. I'll print those and turn off the warning spam first.

```python
import warnings, inspect
warnings.filterwarnings('ignore')
for fn in ['gated3','pair','fit_ro5','fit_ro4','head_run','eval_edit','truth_rows','shift_stats','lda2','lda_in','gget','ols','wc_draws','nll_of','perp','gel']:
    f = globals().get(fn)
    if f is None: print(fn,'missing'); continue
    try: sig = str(inspect.signature(f))
    except Exception as e: sig = 'ERR'
    print(fn, sig[:200], '|', (f.__doc__ or '')[:120])
```