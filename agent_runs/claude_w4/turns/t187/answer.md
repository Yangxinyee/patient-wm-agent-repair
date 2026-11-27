Run 160 shows the original model treats a glucose shift as almost permanent. When both its prediction and the reading move by Δ, the next prediction moves about 0.96Δ (MAP about 0.97). An observation-only shift carries about 0.55Δ. The true insulin response decays to near zero within about nine hours. So an injection along the model's own glucose-level direction will over-persist in the rollout. The gain will have to be calibrated with the forward tool against the 12-hour average, and some shape error at the hour-6 and hour-16 starts is likely.

Next I'll record this in the notebook. Then I'll check that my CPU replica `step_full` reproduces the next state at drug bundles and accepts a perturbation of E. That lets me compute how an added vector in the prelude output E moves the state.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 156-160 (restart 5)
- Model persistence (fwd_2 factual, pred_{t+1} ~ obs_t + pred_t): finger c_obs .554 c_pred .403 (sum .958); art .644/.323 (.966); cuff .39/.57.
  => a belief-level shift persists ~0.96/h in rollout (true insulin IRF decays by lag ~9). Injection gain must be calibrated by forward
  dose_edit against truth (-12.96/+6.48 ins, +4.94/-2.47 NE on sel100); shape error expected for start 6/16.
- nuG/nuM (4,192): per-mg/dL (per-mmHg) direction in new slots y from reading surprise (same in all slots, norm ~1.02 / 5.3 per slot).
- PLAN v2: prelude FFN unit pairs at drug bundles add delta*(gate)*(f(rate)-E[rate|state]); choose delta so dy ~ -g*s*nuG/0.554.
""")
for nm in ['E0','E1','EI','EN','E_','E_nd','E_ref_d','E_ref_n','En','htf','htf_n','hinit','hprev','rws','dEd','dEn']:
    v = globals().get(nm, None)
    if v is None: continue
    print(nm, type(v).__name__, tuple(getattr(v,'shape',())) if hasattr(v,'shape') else (len(v) if hasattr(v,'__len__') else v))
```