Repurposing cost now has a clear lead. h dims are very unequal in scale: median rms 9, but the 90th percentile is 134 and the maximum 217. Most LayerNorm variance comes from a few large dims.

Several dims have rms around 2.5–3.5 (1, 17, 4, 132, 33, 36, 2, 101, …). After LayerNorm they contribute only a few percent of a unit. The old role dims were mostly bigger (rms 6–13), picked on noisy single-dim costs. Choosing the smallest dims, and using a small memory scale (about 1) with a large readout gain, should cut both the information loss and the LayerNorm disturbance.

Before writing the edit code, I need one design fact: whether an insulin start shares a bundle (same minute) with the glucose reading that triggered it. If it does, the prelude sees both and can compute "rate minus policy rate" in one gated write. If not, the policy rate has to come from the state. I'll print the data README (short) and the event timing around insulin starts.

```python
rd=open('/data/README.md').read(); print(len(rd)); print(rd[:1800])
print(tr.columns.tolist()); print(tr.head(3).to_string()[:600])
```