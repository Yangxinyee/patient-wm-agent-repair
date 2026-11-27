At a = −2.6 the main readout is close. On these 554 stay-hours, insulin ×2 is −13.01 against my truth estimate of −13.18 (98% of the needed change). The profile matches through about hour 8 and then overshoots (−20 to −24 at hours 10–11 against −17 to −20). Only the mean is judged, so that is acceptable.

Insulin ×½ under-responds: +5.41 against truth +6.59 (80%). The original model already had the same asymmetry: its ×2 response is 2.4 times its ×½ response, where a linear truth gives exactly 2.0.

Before trying to correct the asymmetry, I need to check my assumption that the true insulin effect is linear in dose. If the simulated effect were curved, the true ×2/×½ ratio would not be 2.0. The within-tier dose multiplier ranges from about 0.5× to 2× in the records, so the training data can test this. I'll look at the one-step training design `Dtr2` and add a squared deviation term.

```python
print(type(Dtr2))
if isinstance(Dtr2, pd.DataFrame):
    print(Dtr2.shape, list(Dtr2.columns)[:40])
elif isinstance(Dtr2, (tuple, list)):
    print(len(Dtr2), [type(t).__name__ + str(getattr(t,'shape','')) for t in Dtr2])
elif isinstance(Dtr2, dict):
    print({k: (type(v).__name__, getattr(v,'shape',None)) for k, v in Dtr2.items()})
print('ins_regress2 args', ins_regress2.__code__.co_varnames[:ins_regress2.__code__.co_argcount], ins_regress2.__code__.co_names[:20])
```