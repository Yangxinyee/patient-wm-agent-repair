The listing came through before the output was cut. It shows the core path the dose can take:
- `z_post.0` (192×576) reads [ctx, e_k, u_k]. Its 64 outputs are the z mean and log-sd.
- `z_in` maps z (32) linearly into a prelude token.
- E reaches h through `b_in` and the recurrent blocks. Then `c_out` (linear) and `coda`, a residual transformer layer over the slots.

So the most direct handle on how a dose updates the belief is the z_post MLP. I could repurpose a few of its hidden units to read the insulin and norepinephrine rate from u_k. Their output would push z in the direction the model uses for glucose or MAP news.

To do that I need to know u_k as a function of rate, and the z_post activation function. My earlier replica attempt left helper functions in the kernel. Next I check whether their source is recoverable and what `act` is, keeping the output short. If `act` is ReLU or GELU, hidden units can be driven into their linear regime for a clean injection. If it is tanh, I need a different plan.

```python
import inspect
for fn in ['act','atom_emb','set_enc2','zpost_variants','meas_atom']:
    try:
        src = inspect.getsource(globals()[fn]); print(fn, 'SRC', len(src)); print(src[:600]); print('---')
    except Exception as ex:
        print(fn, 'no source:', type(ex).__name__)
try:
    tt = torch.tensor([-3.,-1.,0.,1.,3.])
    print('act ->', act(tt))
except Exception as ex:
    print('act call failed', ex)
```