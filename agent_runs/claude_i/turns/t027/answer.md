My estimated truth is far from what the model answers on the same 100 tuning stays:

| edit | model | truth from kernels |
|---|---|---|
| insulin ×2 (finger) | −2.0 | −13.6 |
| insulin ×½ | +1.16 | +6.8 |
| norepinephrine ×2 (arterial) | +0.37 | +4.5 |
| norepinephrine ×½ | −0.06 | −2.2 |

The gap is already there at the first hidden reading. At hour 13, only the hour-12 dose has been edited and all earlier readings are real, so this is effectively a teacher-forced one-step prediction. There the model shifts glucose by −0.74 where the truth is −3.5. So its sensitivity to the latest dose is only about 20% of the true −2 per u/h.

This makes sense for maximum-likelihood training. The random dose component moves the next reading by about 1 mg/dL against a noise SD of 10–13. Ignoring it costs only about 0.001 nats, so the model barely learned it.

Encouragingly, the factual value NLL is just as insensitive. Installing the correct dose sensitivity, centred on the expected dose, should cost almost nothing, and may even help.

Next I need to see what weights are available as levers. I'll list the tensors and shapes and the vocab config/normalisation for the drug codes: z-score centres and scales, and rate and dose bin edges.

```python
print(json.dumps(VOCAB['config'], indent=0)[:2000])
print([k for k in VOCAB.keys()])
for n in NAMES: print(n, SHAPES[n])
```