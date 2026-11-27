The inventory shows I have a full set of local tools:
- a full local run of a stay (`run_stay`) and a multi-step run (`run_from`);
- the local core step;
- prelude attention (`prelude_att`, `prel_heads`) and an attention-edit tester (`att_edit`);
- a gf predicted-mean function (`pred_mean_gf`) and a per-stay evaluator (`eval_stay_rel`);
- `brief` and `info`, helpers I wrote earlier to inspect function code.

The tensor list shows the prelude, `b_in`, `c_out` and the coda all sit inside the `core` group. Only the first 30 core names were shown.

**What I conclude:** I don't need to re-implement anything. I just need to recover what these functions take and return.

**Next step:** print the remaining core tensor names, then use `brief` on the functions that matter for Plan A.

**What would confirm or refute this:** if `run_stay` returns per-bundle states and `pred_mean_gf` gives the head mean from a state, I can test any weight edit locally on whole stays, factually and with doubled insulin inputs, before spending forward passes. If not, I'll build that evaluator next.

```python
core_names = [n for n in NAMES if n.startswith('core.')]
print([n[5:] for n in core_names[30:]])
for f in [run_stay, pred_mean_gf, eval_stay_rel, step_at, att_edit, prelude_att, wc_row, fit_psi, fitread]:
    print('=====', f.__name__)
    try: brief(f, 10)
    except Exception as e: print('brief failed', e)
```