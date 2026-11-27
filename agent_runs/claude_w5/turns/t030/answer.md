Against my estimated truth on these 100 stays, the model's drug effects are 20–2,000 times too small:

| readout | truth (g-estimation) | original model |
|---|---|---|
| insulin x2 | −14.3 mg/dL | −0.55 |
| insulin x1/2 | +7.1 | +0.003 |
| norepinephrine x2 | +4.1 mmHg | +0.19 |
| norepinephrine x1/2 | −2.05 | −0.17 |

The model essentially ignores the dose values when predicting readings.

The model is a complex nonlinear recurrent network, and in the rollout its own sampled readings feed back. So I need to know exactly which tensors carry the dose information and how glucose and MAP predictions are read out before deciding where an edit can act. Next I'll list all tensor names and shapes, to see:
- the structure of `encode_action`;
- the `embed.numeric` columns (z_rate and z_dose);
- the value head (linear or MLP);
- the norms.

I'll also save this comparison to the notebook.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Fwd 1 (dose_edit, original, tuning 100 stays in /work/sel_tuning_A.json): rows = recorded glucose_finger / map_arterial hours 13-24;
  reached = after first edited dose record (>=12). Readout = pooled mean over reached rows.
  original model: ins_x2 -0.545, ins_half +0.003, ne_x2 +0.193, ne_half -0.173.
  g-est truth on same rows (linear superposition of recorded doses from 12): ins_x2 -14.3, ins_half +7.15, ne_x2 +4.11, ne_half -2.05.
  => model nearly ignores doses. Need to inject causal dose effect.
""")
tj = json.load(open('/model/tensors.json'))
print(type(tj), (list(tj.keys())[:5] if isinstance(tj, dict) else tj[:3]))
for n in NAMES:
    print(n, SHAPES[n])
```