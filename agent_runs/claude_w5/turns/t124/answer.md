`test_step` failed on a missing option key (`mem_incl_cur`). The notebook says memory includes the current bundle, so that should be True. `o_best` also gives norepinephrine unit index 3, while the notebook says 4. `enc_bundle` returns (e, u, flag).

Next I find every option key `step` needs by retrying and filling missing keys with guesses, then look at the match error. A small error confirms the local core still reproduces the saved states.

```python
opt_try = dict(o_best); opt_try['unit'] = dict(o_best['unit']); opt_try['unit']['norepinephrine'] = 4
guess = {'mem_incl_cur': True}
asked = []
for it in range(12):
    try:
        r = test_step([sid0], opt_try, kmax=4); print('OK', type(r).__name__, str(r)[:300]); break
    except KeyError as ex:
        k = ex.args[0]; asked.append(k); opt_try[k] = guess.get(k, True)
    except Exception as ex:
        print('ERR', repr(ex)[:300]); break
print('asked', asked); print('global opt:', opt)
```