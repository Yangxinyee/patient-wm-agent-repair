The function list includes `free_run`, `one_step_from_tool`, `scale_atoms`, `ins_atoms_scaled`, `target_readout`, `true_shift_rows` and `eval_full`, so I had begun building rollout and dose-edit tools locally. `SOLU3`, a 192-vector, looks like a correction to the action encoding u that I fitted to match the tool's edited states. The remaining mismatch, about 1e-3, is far above the 1e-7 agreement I had for the original model.

Hour 13 gives a decisive test. The first rolled-out reading there is predicted from the state right after the edited hour-12 drug bundle, so its shift is close to deterministic. Locally the edit should give roughly −1.3 × rate mg/dL there, but the tool shows about +0.06 for stay 2020000016. That points to the rollout's dose-edit input differing from my simulation, rather than an effect that fades over time.

In this run I'll:
- record fwd_4 and fwd_5 in the notebook, so a restart cannot lose them again;
- print the signatures of the key helper functions, so I can run the hour-13 check next.

```python
import inspect
add = """
## Restart 6 (run 250+): RECONSTRUCTED (runs 213-249 were not logged!)
- fwd_4 = dose_edit both, tuning selA (100 stays): ORIGINAL ins x2 -0.545, x1/2 +0.003, ne x2 +0.193, ne x1/2 -0.173;
  EDITED (current edit, kappa_I=-4.6,kappa_N=6.1): ins x2 +0.367 (!), x1/2 -0.244, ne x2 +0.054, ne x1/2 -0.008. => EDIT FAILED IN ROLLOUT.
  Edited factual rollout does move (stay 2020000016 h13 factual 182.8 vs 178.8 orig) but x2 shift at h13 only +0.06.
- fwd_5 = states both, 13 tuning stays (factual). Runs ~230-249 compared local edited states vs tool: mismatch ~1e-3 rel (vs 1e-7 for original);
  bin side (left/right) for rate/dose does not explain it; SOLU3 (192-vec correction to u?) reduces err to 4.6e-4. Unresolved.
- No IPython history available (In=None). LOG EVERY FINDING HERE.
"""
with open('/work/notes.md','a') as f: f.write(add)
for nm in ['free_run','one_step_from_tool','eval_full','scale_atoms','ins_atoms_scaled','target_readout','true_shift_rows',
           'run_cmp','mystep','run_tf','make_bundles','enc2','head_mean','probs_from_v0','v0_of','stay_views','build_edit','gest','gest2','single_err','batch_err','step_with']:
    f_ = globals()[nm]
    try: sig = str(inspect.signature(f_))
    except Exception as ex: sig = '?'
    print(nm, sig, '|', (f_.__doc__ or '')[:150].replace('\n',' '))
for nm in ['tstays','stays_t','stays40','sel_A','selA','sidA','tun','tool_arr','dtool','pm_fwd','pm_loc','SOL','sens_c','sens_e','vbar_I','vbar_N']:
    v = globals().get(nm)
    print(nm, type(v).__name__, (len(v) if hasattr(v,'__len__') else v) if v is not None else None, getattr(v,'shape',''))
```