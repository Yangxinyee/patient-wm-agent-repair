The fwd_2 states file holds 5,294 bundles from 100 training stays, 992 numbers each (`out`, the 4 h slots, the z posterior mean).

The value head is a two-layer MLP (192→192→17) on `with_code`, plus a transition table indexed by the last bin. Doses can only reach it through the state (`out`, `h`, `z`).

My plan follows the g-formula logic. Doses are exogenous given the recorded history, so an edited model whose conditional prediction is right will also give the right counterfactual rollouts. The original model barely uses doses. So the fix is to add a term β·(dose − policy-expected dose given glucose), summed over lags. This should also improve the factual fit rather than harm it.

The cheapest way to add that term is a readout of the state at the head, which needs the state to encode the residual dose history. Before the restart I had started decoding tests (`D`, `cv_r2`), but the results weren't logged. Next I'll recover that code and rerun the decoding for insulin and norepinephrine at lags 1–3.
- High R² would support the head-readout approach.
- Low R² would mean I must build a dose memory in the recurrent core, which needs a local re-implementation of the model.

```python
import inspect
try:
    print(inspect.getsource(cv_r2))
except Exception as e:
    print('no source', e)
print(D.columns.tolist()); print(D.describe().T[['mean','std','min','max']].round(3))
```