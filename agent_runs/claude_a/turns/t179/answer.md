The layer helpers all use their dict argument correctly. So `step_b` receives whatever dict `tf_states`/`tf_eval` hand it, and the substitution must happen in those callers.

I'll test directly:
- does `step_b` give different outputs for the original and `PS` on identical inputs?
- what local names do `tf_states`, `tf_eval` and `tf_loop` use, and is `PS`'s b_in the same object as `P_`'s?

If `step_b` differs between dicts, the callers are swapping the dict, and I'll write my own clean evaluator loop around `step_b`/`head_value` with explicit dicts.

```python
print('tf_states locals', tf_states.__code__.co_varnames)
print('tf_eval', tf_eval.__code__.co_varnames[:30], tf_eval.__code__.co_names[:30])
print('tf_loop', tf_loop.__code__.co_varnames[:30], tf_loop.__code__.co_names[:30])
print('b_in same obj PS/P_:', PS['core.b_in.weight'] is P_['core.b_in.weight'], ' PS==P_ b_in:', bool(torch.equal(torch.as_tensor(PS['core.b_in.weight']), torch.as_tensor(P_['core.b_in.weight']))))
print('id PS', id(PS), 'id P_', id(P_), 'PS is P_', PS is P_)
```