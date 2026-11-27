The state function gives exactly the same states for `PS` and `P_`, yet both differ from the original. So `tf_states`/`tf_nll` apply only the head-side edits shared by both dicts (out_norm, gap, z_head, type_emb, query_norm). They ignore the core weights of the dict they are passed. That makes the +0.006 scaffold "cost" an artefact of out_norm/gap edits, not a measurement of the core scaffold.

My notes say a second evaluator, `tf_eval(P, build_ds(df, stays, 0))`, and the rollout replica `rollout_stay` were validated separately. I'll check whether `tf_eval` responds to core edits: compare the original, `PS` and `P_` on 20 sel2 stays. If `PS` ≠ `P_`, `tf_eval` is a sound factual evaluator and I'll use it from now on.

```python
dfs=[(k,v.shape) for k,v in globals().items() if isinstance(v,pd.DataFrame) and len(v)>100000]
print('big dfs', dfs)
sel=json.load(open('/work/sel.json')); print(type(sel).__name__, (list(sel.keys()) if isinstance(sel,dict) else len(sel)))
```