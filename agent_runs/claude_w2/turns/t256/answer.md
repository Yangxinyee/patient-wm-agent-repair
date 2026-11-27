Run 247 showed there is no input history (`In`/`_ih` are absent), so runs 208–244 can't be recovered beyond the saved scripts. I now know the building blocks: the exact local model, the rollout engine (`run_rollout_h2` with hooks), the exact-shift oracle hook (`make_lin_oracle` with dose-lag weights `a_i` and policy-lag weights `c_i`), and `gen_wc`, which collects head features under changed doses.

My working hypothesis is unchanged: the edit has to add a dose-driven shift in the value head, gated by reading code. It has to read the actual dose from the state, with a policy compensation through the per-(code, last bin) table `embed_transition.value_bin`, so factual predictions don't double count.

To continue without guessing, I need the function signatures and the prepared rollout structures still in memory. This run only prints short facts: signatures and the remaining global names. Errors and short output are expected; nothing is tested yet.

```python
import inspect
fns=['prep_rollout','run_rollout','run_rollout_h','run_rollout_h2','readout','eval_arms','truth_rows','dose_stack','lag_abs',
     'make_lin_oracle','make_dev_oracle','tilt_P','head_value_logits','gen_wc','wc_hidden','lag_feats','collect_states_ins','pfun','true_shift_rows']
for f in fns:
    fo=globals().get(f)
    try: print(f, str(inspect.signature(fo))[:150] if fo else 'MISSING')
    except Exception as ex: print(f,'?',ex)
gl=[k for k,v in globals().items() if not k.startswith('_') and not isinstance(v,types.ModuleType)]
i0=gl.index('umapB') if 'umapB' in gl else 0
print(len(gl), 'names after umapB:', ' '.join(gl[i0:])[:3500])
```