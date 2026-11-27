Run 40 gives the readout definition: per stay-hour shifts from hours 13–24, averaged over the "reached" rows, meaning hours after the drug had run from hour 12. `TR` turns out to be my kernel-implied true shift per stay-hour, already built. For stay 2020000000 under insulin x2, the model predicts about −1.5 to −3.4 mg/dL while the kernel implies −2 rising to −13. Under insulin x1/2 the model barely moves.

Next, I will average `TR` over reached rows per edit and set it beside the model's readouts. I'll also check that my reached flags match the tool's. If the kernel truth is several times the model's shift for all four edits, the diagnosis is "too little causal dose response", and I need the amplification factors per drug.

```python
m = TR.merge(de1, on=['stay_id','hour','edit'], how='inner')
print(len(m), (m.reached_t==m.reached).mean())
for e in ['insulin_x2','insulin_half','ne_x2','ne_half']:
    q = m[(m.edit==e)&(m.reached)]
    print(e, "n", len(q), "model %.3f"%q['shift'].mean(), "kernel-true %.3f"%q['true'].mean(),
          "ratio %.2f"%(q['true'].mean()/q['shift'].mean()))
    # by hours-since-12 profile
    print("   by hour model:", np.round(q.groupby('hour')['shift'].mean().values,2))
    print("   by hour true :", np.round(q.groupby('hour')['true'].mean().values,2))
```