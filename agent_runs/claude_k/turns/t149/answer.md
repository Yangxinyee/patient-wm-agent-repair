The probe route doesn't fit the factual budget. Even the factual-optimal gain (λ = 1) slightly worsens the out-of-fold NLL for insulin (+0.0006). At λ = 1 the counterfactual response is only −0.44 mg/dL per unit/h of sustained insulin, against roughly −3.4 needed for the direct effect alone. Amplifying costs too much:

| Drug | λ | ΔNLL per reading |
|---|---|---|
| Insulin | 2 | 0.0034 |
| Insulin | 3 | 0.0085 |
| Norepinephrine | 5 | 0.004 |

The model's q simply doesn't hold a usable, dose-specific signal. That is a limit of the information in the state, not of the probe.

So the dose has to be written into the state more distinctly at the drug bundle itself, where it is clean (u_k comes straight from the drug records). The value head can then read that channel with the code-gated paired units. Part of the dose is implied by the readings, and the head can subtract that part exactly through the last-value-bin embedding or the code×last-bin table. That leaves a residual-based correction in factual mode and a full response to dose changes under edits.

Per the architecture, u_k reaches the state only through the prelude layer (slots attending to the u_k token, or to the token built from z_post). Next I'll list the core, prelude, action-encoder and z tensors to find where a dose-proportional term can be injected with predictable reach.

If the prelude attention has separable value and output projections, I'll design a rank-1 injection there and measure its effect on states with the tool before reading it out.

```python
for pre_ in ['core.','encode_action','z_post','z_in','prelude']:
    print(pre_, [(n,SHAPES[n]) for n in NAMES if n.startswith(pre_) and 'blocks' not in n][:40])
print('blocks', [(n,SHAPES[n]) for n in NAMES if 'blocks.0.0' in n])
```