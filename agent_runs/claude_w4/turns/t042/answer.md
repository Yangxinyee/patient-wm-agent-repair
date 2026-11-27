The architecture explains why the dose answers are weak: dose reaches glucose and MAP predictions only through the action vector u_k, which feeds the z posterior and the prelude.

The original model's answers on 100 tuning stays are almost zero:

| Edit | Model's predicted shift |
|---|---|
| insulin ×2 | −0.35 mg/dL |
| insulin ×½ | +0.21 mg/dL |
| norepinephrine ×2 | +0.02 mmHg |
| norepinephrine ×½ | −0.25 mmHg |

The records say the effects are much larger. Insulin lowers glucose by about 2 mg/dL per unit/h after one hour, about 10 cumulatively over 12 hours. Norepinephrine raises MAP by about 3.7 mmHg per mcg/kg/min after one hour, about 10 cumulatively. So the model has largely learned how doses and readings go together, not what the drugs do.

Before choosing an edit I need the exact tensor layout of the action encoder, z_post, prelude, core and value head. A CPU copy of the model might be feasible: the fwd_3 states give exact out/h/z values to check it against. A working copy would let me run unlimited rollouts on training stays without spending forward passes. I will print the tensor shapes, the vocab config, and my earlier `true_shift` estimate for the 100 tuning stays. If the layers are standard, I will start the CPU copy; if not, I will fall back to editing based on regressions over states.

```python
tj = json.load(open('/model/tensors.json'))
print(type(tj), (list(tj.keys())[:5] if isinstance(tj, dict) else tj[:3]))
for nm in NAMES:
    print(nm, tuple(SHAPES[nm]))
print(json.dumps(VOCAB.get('config', {}))[:1500])
print('true_shift:', true_shift if not hasattr(true_shift, 'shape') else (getattr(true_shift,'shape',None), true_shift))
```