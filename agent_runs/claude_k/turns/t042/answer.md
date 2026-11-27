The architecture shows exactly one route for a dose to reach the predictions. Insulin and norepinephrine atoms are encoded by `encode_action` into u_k. From there, u_k enters only `z_post` and the prelude token. It then has to pass through the prelude transformer, the recurrent block, the coda and `out_norm` before the heads read it.

The value heads cannot see the dose directly. Their "last" register holds the same code's previous value, so the glucose head sees only the last glucose, never the insulin rate. The dose effect must therefore come from the state, through view → q → with_code → `value_bin`.

My working hypothesis is that the model badly underweights the dose. In fwd 1, insulin ×2 gave -2.15 against a true effect of about -13.8, and norepinephrine ×2 gave +0.34 against about +4.6. Because the records are confounded, I expect the model has partly absorbed the missing effect into how it reads glucose and MAP levels.

That points to a correction of the form (true kernel − model kernel) × (dose − policy-expected dose given readings). In factual data this correction averages to zero, so it should not hurt, and could even help, the factual NLL. It also fixes the counterfactual response.

To build it I need the exact module structure. In particular I need to know whether `value_bin` is a linear layer or an MLP, and the shapes of `embed_transition`, `embed.numeric` and `encode_action`. Those decide where a code-gated, linear dose term can be inserted. This run only gathers that information, so it cannot confirm or refute the hypothesis yet.

```python
import json
print("; ".join(f"{n}:{tuple(SHAPES[n])}" for n in NAMES))
print()
print(json.dumps(VOCAB['config']))
print(list(VOCAB.keys()))
print({k: (VOCAB[k] if not isinstance(VOCAB[k], (dict, list)) or len(str(VOCAB[k])) < 300 else str(VOCAB[k])[:300]) for k in VOCAB if k != 'config'})
```