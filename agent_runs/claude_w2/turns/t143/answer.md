The kernel has head functions (`head_probs`, `head_pm`, `eval_wc`, `score`), and the last stored searches were about the value head. Their keys are the `last_numeric` column assignment (`assign`), the status options (`stat_opts`) and the age/gap modes. The best head score so far was ~2.1–3.3, with correlation 0.9989 against the tool's predicted means. So by run 133 the head reproduced the tool's predicted mean closely but not exactly. I'll recover the source of the latest functions (`build_mem`, `head_probs`, `head_pm`, `eval_wc`, `score`) to see exactly what is implemented and what error is measured. Then I'll decide whether to finish the exact local rollout or switch to a simpler, tool-verified edit.

```python
import inspect
for fn_ in ['build_mem','head_probs','head_pm','pm_from','f8_from','eval_wc','score']:
    try:
        print(f"----- {fn_}"); print(inspect.getsource(globals()[fn_])[:1800])
    except Exception as ex:
        print(fn_, 'no source', ex)
print('In' in globals(), '_ih' in globals())
```