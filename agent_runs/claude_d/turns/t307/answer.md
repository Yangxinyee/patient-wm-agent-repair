E5 moved ne_x2 only from +1.03 to +1.35; the target is +4.85. The other three readouts are unchanged: insulin x2 −13.91 (target −13.99), insulin x1/2 +7.08 (target +6.99), NE x1/2 −2.16 (target −2.43). That is as expected, since E5 only touches out-of-range NE atoms.

So the beta=1 linear extrapolation of the out-of-range NE representation is read by the model, but weakly. It delivers about 0.3 of the 3.97 mmHg out-of-range part. If the response stays linear, beta around 12 would close the gap. It could also saturate.

Before rebuilding I need two things:
- The call signature of `build_enc` and what it uses (its source can't be shown, but signature and referenced names can).
- A reminder of how the NE effect was estimated, in particular whether linearity in rate was checked. Most of the ne_x2 target comes from extrapolating beyond the recorded cap of 1.0, so I want to be sure that assumption came from the records.

This run gathers both.

```python
import inspect
for fn_ in [build_enc, build_ne2, u_from_m, solve_u, test_u, du_for, gate_f, uenc2, set_enc, m1, m1_bf, true_shift_ne, eval_edit]:
    try: print(fn_.__name__, inspect.signature(fn_), fn_.__code__.co_names[:30])
    except Exception as ex: print(fn_.__name__, 'ERR', ex)
txt = open('/work/notes.md').read()
print(len(txt))
L = [l for l in txt.split('\n') if any(k in l for k in ['NE','norep','linear','cap','satur','Emax','gam_ne','effect'])]
print('\n'.join(L)[:3500])
print(open('/work/effects.json').read()[:1200])
```