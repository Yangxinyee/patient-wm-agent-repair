The bug is clear. `lin` was overwritten by a one-argument lambda, but `mlp2` and `q_from_rows` call `lin(x, name)`, a linear layer that loads `name.weight` and `name.bias`.

I'll keep the lambda as `lin_r` and restore `lin(x, name)`. Then I check that `eval_edit(gated3(0,0))` reports zero cost, which confirms the reconstruction matches the stored original head outputs. After that I evaluate `gated3(-1.9,3)` offline and compare with the measured tuning cost of +0.0034 per reading.

```python
lin_r = lin
def lin(x, name):
    W = T(name + '.weight')
    y = x @ W.T
    if (name + '.bias') in SHAPES:
        y = y + T(name + '.bias')
    return y
import time
t0 = time.time()
r0_ = eval_edit(gated3(0.0, 0.0), 'zero', True)
print('---- t', round(time.time()-t0,1))
r1_ = eval_edit(gated3(-1.9, 3.0), 'g3 -1.9/3', True)
print('---- t', round(time.time()-t0,1))
print(type(r1_), (list(r1_.keys()) if isinstance(r1_, dict) else r1_))
```