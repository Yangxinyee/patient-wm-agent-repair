The core is a pre-norm prelude layer, two recurrent block layers and a pre-norm coda layer. All of them have residual paths and LayerNorms, and the state passes through `c_out` and the coda every step. A dedicated memory dimension would therefore need its writes zeroed in each residual output and its reads removed from each norm.

Before designing that, I need to know whether my local `step` reproduces the GPU states. I'll check the constants used by the latest checker (`step_err3`) to find its mode options, then run it on its default stays. If the error is small (relative error under about 1e-3), the local model can be used for designing the edit.

```python
c = step_err3.__code__
print([x for x in c.co_consts if isinstance(x, (str, int, float)) and not (isinstance(x, str) and len(x) > 40)][:40])
print([x for x in step.__code__.co_consts if isinstance(x, (str, int, float))][:40])
```