This explains why the model cannot see the dose. Before encode_action's final LayerNorm, the vector v is dominated by a single direction whose size swings enormously with the rate: its standard deviation runs from 6 to 469, mean 225 for insulin-only bundles. Only 4 directions are significant.

The LayerNorm divides that size away, so u collapses to roughly two saturated states, one for each sign of that component. That is the binary pattern I saw in z dims 9 and 20, and it is why u carries the rate only weakly.

So no small addition to v can carry the rate linearly: its gain would vary about 80-fold across bundles. The encoder must be rewritten so that its normalisation sees a large, fixed component (making it near-linear). The original content would be kept at a small scale, with clean linear insulin and norepinephrine rate terms on top. That is only acceptable if predictions barely depend on u's current content.

Next I will reuse my existing local head functions (head_inputs, pred) to measure that dependence. First I need their source and signatures.

- **Confirms:** replacing u leaves the predicted glucose and MAP means nearly unchanged. Then rewriting the encoder is safe.
- **Refutes:** the predictions move noticeably.

```python
import inspect
for fn in [head_inputs, pred, head_feats, pred_v]:
    try:
        src_=inspect.getsource(fn); print('-----', fn.__name__, len(src_)); print(src_[:1500])
    except Exception as ex: print(fn.__name__, 'no source', ex)
```