The column-0 search found almost no function definitions in impl3.py, so the functions are probably indented or defined another way. I also don't remember exactly what XE and XU (620×192) hold; I think they are E and u at drug bundles. Next I'll list every `def` line in the three impl files with its signature and the first comment lines of impl3.py. That shows how to call `core_step`/`run_stay` without re-reading 13k characters. This is still a lookup step.

```python
for fn_ in ['/work/impl.py','/work/impl2.py','/work/impl3.py']:
    s = open(fn_).read()
    print('=====', fn_, len(s))
    for i, line in enumerate(s.splitlines()):
        if 'def ' in line or line.strip().startswith('#'):
            print(i, line[:160])
import inspect
for f_ in ['core_step','run_stay','step_at','head_inputs','pred_mean_gf','enc_eu','stay_data','step_parts','core_step_v']:
    try: print(f_, inspect.signature(globals()[f_]))
    except Exception as ex: print(f_, 'ERR', ex)
```