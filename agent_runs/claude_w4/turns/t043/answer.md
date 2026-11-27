The tensor list gives the full structure: standard-looking pre-norm attention and FFN layers with LayerNorm-style biases, 4 heads, and a two-layer `z_post` MLP over [ctx, e_k, u_k]. The value head is a two-layer MLP over the query plus a register term.

This shapes the edit. To be right both in factual mode and under dose edits, the model's glucose prediction should shift by the effect of the dose surprise: the recorded dose minus the dose the sliding scale would have given. The old model already folds the expected dose into its glucose dynamics. Adding the raw dose effect would bias factual predictions by several mg/dL and weaken the counterfactual answers.

Building that pathway through attention and norm layers blindly would waste forward passes. A CPU copy of the model would let me measure linear responses and run rollouts on training stays locally. The fwd_3 states give exact targets to check it against step by step.

Next I will:
- recall how my `true_shift` and `tsls` functions work, to confirm the impulse responses hold other doses fixed;
- compute the true readouts for the 100 tuning stays used in fwd_1;
- inspect vocab.json for the tokenisation details needed for the copy.

If the true insulin ×2 shift on those stays is large (around −10 to −20 mg/dL), the old model's −0.35 confirms it is badly off.

```python
import inspect
for fn in ['true_shift', 'tsls', 'crossfit']:
    try:
        print('=====', fn); print(inspect.getsource(globals()[fn]))
    except Exception as ex:
        print(fn, 'source unavailable:', ex)
print(type(VOCAB), list(VOCAB.keys()))
for k, v in VOCAB.items():
    s = json.dumps(v)
    print('--', k, len(s), s[:700])
```