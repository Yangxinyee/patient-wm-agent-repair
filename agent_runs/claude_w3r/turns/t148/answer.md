Source isn't recoverable: the kernel functions were defined through exec, so `inspect` can't find their text. I'll treat them as black boxes and recover their usage from their code objects (argument names, globals used, return statements). If that proves too opaque, I'll rewrite a clean core step into a file from the architecture details in my notes. Success means I can call `stay_data` and `run_stay` correctly on one stay.

```python
import dis, io, contextlib
def brief(f, maxlines=60):
    c = f.__code__
    print('==', f.__name__, 'args', c.co_varnames[:c.co_argcount], 'defaults', f.__defaults__)
    print('   globals used:', c.co_names)
    print('   locals:', c.co_varnames[c.co_argcount:])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): dis.dis(f)
    lines_ = [l for l in buf.getvalue().splitlines() if 'RETURN' in l or 'BUILD_TUPLE' in l]
    print('   return-ish:', lines_[:6])
for f_ in [stay_data, run_stay, step_parts, step_at, enc_eu, head_inputs, pred_mean_gf]:
    brief(f_)
```