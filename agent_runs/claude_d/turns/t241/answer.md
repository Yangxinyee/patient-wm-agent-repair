The architecture shows where norepinephrine information can enter.
- A drug atom's rate enters an action token u_k, through a linear z_rate column and fine-bin lookups.
- Slot tokens read u_k by attention in the prelude, and the result reaches h and out.
- The value head reads the state only through the view: out, the mean of the decayed slots, and a prior-sampled z.
- For a MAP reading, the head's "last" register holds only the last MAP value, with no drug information.

So, as with insulin, the head can learn about norepinephrine only from what the state retains. Earlier decoders found that weak.

Before choosing between a head-level norepinephrine channel and an input-side change, I need two things:
- The column layout of my reading table, and the signatures of the helpers I built earlier, so I can reuse them without wasting runs.
- Later, how often a MAP reading directly follows a norepinephrine bundle, where the information should be strongest.

```python
import inspect
print(list(RD2.columns)); print(list(STALL.columns))
for f in ['build_ins3','eval_E','eval_edit','head_mean2','wc_from','aug_wc','cvdec_aug2','fit_tilt','exp_tilt2','lin_gate','ridge_gate','ctrl_ne','f_ne','true_shift_ne','solve_u','test_u','variant','head_inputs','mean_from_A']:
    try: print(f, inspect.signature(globals()[f]))
    except Exception as e: print(f, 'ERR', type(e).__name__)
```