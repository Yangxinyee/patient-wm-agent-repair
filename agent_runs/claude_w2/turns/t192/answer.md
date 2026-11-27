Run 186 recovered the main quantities from the undocumented runs:
- **Insulin kernel:** the effect of a one-hour pulse of 1 U/h on glucose k hours later is -2.0, -2.5, -2.2, -1.8, -1.4, -1.15, -0.94, -0.75, -0.50, -0.42, -0.33, -0.27 (Kins).
- **Norepinephrine kernel (Knor):** +3.6, +2.7, +1.7, +1.1, +0.4, +0.45, about 0 after that.
- **Propagation factor:** kap = 0.42, used to turn the true kernel into the direct shift needed at the model's head.
- **Arm sets** I had prepared, including random per-infusion multipliers for fitting decoders across dose levels.

The oracle results themselves were never written to the notes.

My current hypothesis is a head-level edit that shifts the predicted glucose (or MAP) mean by Σ_k G_k·(dose − policy dose). It uses the deviation from the policy, not the raw dose, so the factual fit is kept and the policy effect already learned through glucose levels is not counted twice. It is confirmed if a model-plus-oracle rollout reproduces my record-based effects (about -14 for insulin x2 on the 100 stays of sel1), and refuted if the readouts stay far off for every reasonable kernel.

Next I need three things: the rest of make_dev_oracle (its print was cut), the readout function, and the name of the training-record dataframe. I'll also add the recovered state to the notes now so a restart cannot lose it again.

```python
s=open('/work/oracle.py').read(); i=s.find('def make_dev_oracle'); print(s[i:])
s2=open('/work/rollout.py').read(); i2=s2.find('def readout'); print('-----readout-----'); print(s2[i2:i2+2500])
print('-----dataframes-----')
for k_,v_ in list(globals().items()):
    if isinstance(v_,pd.DataFrame) and 'event' in v_.columns: print(k_, v_.shape, v_.stay_id.nunique())
notes_add="""
## Restart 6 (runs 183-187): recovered undocumented work (runs ~150-182)
- Files: tf.py teacher_forced(df,ids,W) (local factual); cf.py arm_embed/arm_mult/dose_grid(R,drug,mult)->G[si,h+3]; cf_collect(df,ids,arms,W) stores head inputs;
  oracle.py: tilt_P(P,cen,r) exact mean-shift tilt; make_oracle(R,DGi,Gi,...) tilt by arm-dose-difference; make_dev_oracle(R,DGi,Gf,t_hide,pf) tilt by sum_k Gf_k*(dose-policy(glucose)) on current path;
  G_from_K(K,kap): direct kernel G[m]=K[m]-kap*sum(G[:m]); truth_rows(R,DG,a,rows,Kf). policy.py: build_policy_table(df), make_pf_band(tab) -> pf(glucose).
  rollout_h.py: run_rollout_h(R,W,arms,...,hook_P,hook_draw,hook_state).
- Kernel vars: Kins (12 lags) -2.01 -2.49 -2.24 -1.80 -1.42 -1.15 -0.94 -0.75 -0.50 -0.42 -0.33 -0.27; Knor 3.60 2.73 1.72 1.08 .42 .45 0 .2 .14 .05 0 0; kap=0.42 (meaning not recorded).
- Oracle results NOT recorded -> redo, and record every result immediately.
"""
open('/work/notes.md','a').write(notes_add)
```