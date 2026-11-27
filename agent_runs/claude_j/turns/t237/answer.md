Run 209 gave the signatures. `frontier(wf, alpha)` fits the ridge readout out-of-fold with factual weight `wf`, applies the tilt, and scores it with `dnll_groups` and `summarize`. So "expF" is probably the factual log-loss change of the edit. If so, even the strongest penalty (+0.02) is above the 0.005 limit.

Next I will read the string constants and local variable names of `frontier`, `dnll_groups` and `summarize` to confirm what expF measures. If it is a factual log-loss change, then a head-only readout cannot meet the limit and I need a better dose signal. If it is something else, such as an explained fraction, the head-unit plan may still work.

```python
for n in ['frontier','dnll_groups','summarize','fit_eval']:
    c=globals()[n].__code__
    s=[x for x in c.co_consts if isinstance(x,str)]
    print(n,'| C:',s[:12],'| V:',c.co_varnames[:20])
```