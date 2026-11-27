The dose_edit file gives, per stay, hour and edit, the model's shift, its factual mean and a "reached" flag. `true_shift_ins` is a helper function in the kernel. I'll print its source and the shapes of the tuning-stay panels. Then I can line up the true and model shifts hour by hour on the same reached stay-hours.

```python
import inspect
try: print(inspect.getsource(true_shift_ins))
except Exception as ex: print('nosrc', ex)
print(INS_U.shape, NOR_U.shape, GF_U.shape, type(sid2i), len(sid2i), len(sel1), type(stU), getattr(stU,'shape',None))
print(beta_ins if not callable(beta_ins) else 'fn')
```