Reading the lag-1 insulin innovation from the value head's inputs works only weakly: CV R² is 0.24 from the view and query, and below zero from the hidden layer. A readout gain large enough to hit the target would add roughly 0.007 nats of factual error, over the 0.005 limit. So the dose signal has to be strengthened upstream in the recurrent core.

Only the value head and the core matter here. The rollout keeps the recorded event times and codes and draws only values, and the factual criterion counts only value NLL.

To design an upstream edit I need a local one-step replica of the core: previous state plus one bundle gives the next state. I can validate it bundle by bundle against the returned states. The first check is the z posterior mean, computed from context plus the bundle encodings. That needs the raw event rows, so I'll look at the tuning file's format for one stay.

```python
print(tu.shape, list(tu.columns))
s0=tu.stay_id.iloc[0] if 'stay_id' in tu.columns else None
print(tu[tu.stay_id==s0].head(12).to_string()[:1500])
```