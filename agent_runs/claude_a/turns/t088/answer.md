The full offline model now runs end to end from the records alone. Across all 5,369 bundles of the 100 sel2 stays, the error stays at 0.011 on h (scale 35) and 1.3e-4 on out, and the pass takes 4 s. I can therefore compute any stay's state offline and roll it forward.

Before writing the rollout I need three of the tool's conventions:
1. Which stay-hours carry a readout row: every hour 13–24, or only hours with a recorded reading.
2. How "reached" is defined.
3. Whether readout hours without a recorded glucose_finger reading still get a prediction.

I'll compare the forward-1 rows for a couple of stays with their tuning records. The check: if "reached" starts at the first hour after the first infusion that starts at hour ≥ 12, my reading of the rule is right.

```python
with open('/work/notes.md','a') as fh:
    fh.write("- Run82-83: first bundle: h0=0,out0=0,dt0=1s. Batched replica (ds_b, core_step_b, ctx_fn, zpost_mean, tf_stay) self-driven TF over sel2: max|dH| 0.011, |dOut| 1.3e-4. FULL OFFLINE MODEL WORKS.\n"
             "- Effects shape (runs 80-81): insulin lag1 slope per unit/h ~-1.94..-2.17 across tiers (G 165..266) => additive, linear (quad~0). NE lag1 slope per unit ~3.0-3.5 across N_prev 0.1..1 => linear in rate (not log).\n"
             "- Targets on sel1 reached rows (my psi): ins_x2 -14.30, ins_half +7.15, ne_x2 +4.08, ne_half -2.04 (orig model -1.88,+0.97,-0.41,-0.52).\n")
for s_ in [2020000000, 2020000001]:
    g_ = D1f[(D1f.stay_id==s_)&(D1f.edit=='insulin_x2')][['hour','shift','factual','reached']]
    rec = tu[(tu.stay_id==s_)&(tu.hour>=11)&(tu.hour<25)]
    print("stay", s_, "insulin starts >=11:", rec[rec.event=='insulin'][['hour','rate']].values.tolist())
    print(" gf hours >=13:", rec[rec.event=='glucose_finger']['hour'].tolist())
    print(" D1f hours:", g_['hour'].tolist(), "reached:", g_['reached'].astype(int).tolist())
cnt = D1f[D1f.edit=='insulin_x2'].groupby('stay_id').size(); print("rows per stay (ins):", cnt.value_counts().to_dict(), "n stays", len(cnt))
cntn = D1f[D1f.edit=='ne_x2'].groupby('stay_id').size(); print("rows per stay (ne):", cntn.value_counts().to_dict(), "n stays", len(cntn))
```