`step_err3` takes a memory mode, and 'empty_const' is one option (action-only bundles store a constant empty vector in the memory M). `step` has a variant switch for the age encoding ('age_s', 'agemode', 'add_mem'). I'll run `step_err3('empty_const')` on its default 4 stays. If the relative error of h and out against the GPU states is about 1e-3, the local core is validated. A large error means the memory or age handling is still wrong.

```python
import time
t0 = time.time()
r_ = step_err3('empty_const')
print(type(r_), (r_ if not hasattr(r_, '__len__') or len(str(r_)) < 1500 else str(r_)[:1500]), round(time.time()-t0, 1))
```