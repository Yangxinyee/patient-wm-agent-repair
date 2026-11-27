On the 100 tuning stays, the true effects (linear SNMMs, same reached stay-hours) are:

| edit | true | model |
|---|---|---|
| insulin ×2 | −14.0 | −1.73 |
| insulin ×½ | +7.0 | +0.91 |
| NE ×2 | +4.85 | −0.08 |
| NE ×½ | −2.43 | −0.48 |

So the model is far too weak on insulin and wrong in sign or size on NE. The edit needs to roughly add the missing direct dose effect to the glucose and MAP predictions.

The plan I'm weighing works inside the value-head MLP. I would repurpose pairs of hidden units into code-gated linear units, active only for glucose codes (or only for MAP codes). Each pair adds a logit tilt proportional to a dose-history feature read linearly from the head input. Using a pair of units cancels the gating offset.

Designing and checking that needs the head's real inputs: out and h from the states tool, plus exact bundle times for dt. So first I'll look at the record format (the README and tuning.csv), and at the code, event and atom vocabularies, to rebuild bundles and register fields.

```python
print(open('/data/README.md').read()[:2500])
print(tu.head(8).to_string())
print('codes', VOCAB['codes'])
print('event_kinds', VOCAB['event_kinds']); print('atom_kinds', VOCAB['atom_kinds'])
```