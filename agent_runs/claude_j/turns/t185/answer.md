The z posterior matches the tool closely when dt is in seconds (median error about 0, max 0.03). The replica is now complete for counterfactual work:

- start from the tool's factual state;
- recompute u_k for edited drug atoms;
- recompute z with `zpost`;
- step with all bundles in memory, ages in seconds.

My plan:
1. Generate teacher-forced counterfactual states: readings as recorded, doses scaled from hour 12.
2. Compute the head input q at each glucose/MAP prediction, and the model's own predicted-mean change.
3. Set the target as the structural direct effect of the dose change given the readings, minus what the model already does.
4. Fit a ridge readout from q to that target, and implement it in a dead value-head unit.
5. Calibrate with the dose_edit tool.

First I will save these findings to the notes and print my head replica (`/work/heads.py`, 2.3k chars) so I can reuse it.

```python
with open('/work/notes.md','a') as f:
    f.write("""- Run 161-163: replica validated: step(out_p,h_p,dt_s,e,u,z,Mem,ages): Mem = e of ALL bundles 0..k (incl current, last 128), ages seconds; one-step out err med .01 (GPU precision).
  zpost: ctx=out_p+out_norm(mean_slots(flow(h_p,dt)))+gap(log(dt_s)/10); z=z_post.2(gelu(z_post.0([ctx,e,u])))[:32] -> err ~0 (dt seconds). vecs(s,k)=bvd(BUN[s][k][1]).
- PLAN: teacher-forced cf states (doses scaled from h12, readings recorded) -> target = structural direct effect given readings - model change -> ridge readout on q -> dead value unit; calibrate via dose_edit.
""")
print(open('/work/heads.py').read()[:2400])
```