impl3.py turns out to be mostly a dump of the sklearn source plus the edit helpers. The working local core implementation (`core_step`, `run_stay`, `step_parts`, …) exists only in the live kernel. That is fragile across restarts. So next I'll save the source of all my kernel functions to /work/impl4.py, and print how `stay_data`, `run_stay` and `step_parts` are used, so I can compute E at drug bundles for the probe. Success means the file is written and the usage is clear.

```python
import inspect
names_ = ['fbin','zsc','m_of','build_bundles','atom_vec','enc_eu','LNt','mha','ffn','mlp_t','age_emb','core_step','core_step_v','core_step_m','core_step_p',
          'stay_data','run_stay','step_at','step_parts','head_feats','head_inputs','pred_mean_gf','pred','pred_c','pred_v','u_of','s_of','fn','sv','score','score2','eval_stay_rel']
out_src = []
for n_ in names_:
    try: out_src.append(inspect.getsource(globals()[n_]))
    except Exception as ex: out_src.append(f"# {n_}: no source ({ex})\n")
open('/work/impl4.py','w').write("\n".join(out_src))
print(len("\n".join(out_src)))
for n_ in ['stay_data','run_stay','step_parts']:
    print(inspect.getsource(globals()[n_])[:1500]); print('-----')
```