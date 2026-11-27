The model's response is too weak from the first hour, not only in how it builds up. At hour 13, which reflects only the doubled hour-12 dose, the model shifts glucose by -0.55 against a true -2.6. For norepinephrine it is +0.07 against +1.8. After that the model's shift grows slowly, staying about 5 times too small for insulin and 8-25 times for norepinephrine. So the direct dose-to-next-reading pathway is too weak in rollouts, at least for doses above the usual range.

The most powerful tool would be an offline re-implementation of the full step, letting me test edits without spending forward passes. Earlier I seem to have started reproducing the core step (z posterior, prelude). Next I'll list the functions defined in the kernel and their first lines, to see how complete that reconstruction is. If a step function exists and matches the fwd_3 states, I can simulate rollouts offline; if not, I'll pick a simpler editing strategy.

```python
import inspect, types
fn = {k: v for k, v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_')}
print(len(fn))
for k, v in fn.items():
    try:
        src = inspect.getsource(v); print(k, len(src), '|', src.split('\n')[0][:110])
    except Exception as e:
        print(k, 'nosrc')
```