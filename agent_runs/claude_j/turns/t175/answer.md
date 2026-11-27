The vocab and embedding tables show that drug atoms take many embedding terms (kind, ekind, categorical unit/rate_unit/route/action/status tables, kq). This is where the earlier replica work got stuck. The weight dicts and states arrays I relied on (W, Wt, BUN, Xt) are no longer in the variable list.

Before deciding between a full replica (needed for teacher-forced counterfactual states) and a simpler head-level readout fitted on the tool's factual states, I will check what the existing replica functions depend on. I will print the global names each key function references, which shows which weight dicts and helpers they need and whether they still exist. The output stays short.

If they reference globals that no longer exist, I will rebuild the replica cleanly in a file. If they are intact, I can test them on drug bundles straight away.

```python
for k in ['step','encode','build_bundles','bundle_vecs','drug_atom2','meas_atom','head_mean','view_parts','q_for','true_shift','rows_truth']:
    c=globals()[k].__code__
    gl=[n for n in c.co_names if n in globals() or n not in dir(__builtins__)]
    print(k, [ (n, n in globals()) for n in gl][:14])
```