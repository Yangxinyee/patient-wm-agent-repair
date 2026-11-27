The dose_edit file gives a shift per stay-hour plus a `reached` flag, and the summary means come from the reached rows. So I can compute my impulse-response "truth" on exactly the same stay-hours.

I lost the source of `true_shift`, so I will rebuild the truth calculation transparently. First I need three things: the event table's column layout, the impulse-response vectors stored in the kernel, and whether the summary's mean is pooled over stay-hours or averaged per stay.

```python
print(tr.columns.tolist()); print(tr.head(3).to_string())
for nm_ in ['h','hF','hA','hL12','bF','bL','bA','seF','seA']:
    v = globals().get(nm_)
    if v is not None: print(nm_, np.round(np.asarray(v),2))
for ed_ in de1.edit.unique():
    q_ = de1[(de1.edit==ed_) & de1.reached]
    print(ed_, 'pooled', round(q_['shift'].mean(),3), 'stay-mean', round(q_.groupby('stay_id')['shift'].mean().mean(),3), len(q_))
```