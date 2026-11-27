The dose and rate embeddings are as large as the code embeddings: bin-row norms are 0.3–2.7, and the z_dose and z_rate columns of `embed.numeric` have norms of about 3.1 and 3.2. So the action encoder receives plenty of dose information, and the original model mostly learned to ignore it for readings. That fits with the records, which show no causal effect.

The blinding edit makes every infusion look like a median infusion:

- Zero the z_dose and z_rate columns of `embed.numeric`. Only drug atoms carry nonzero values there, and z = 0 at the median.
- Set rows 1–16 of `embed.bins.dose` and `embed.bins.rate` to row 8. That is the bin of the median dose and rate for both drugs.

This should make every dose edit produce identical model states, so all four shifts become exactly 0. The only cost is losing the small information the dose level carries for factual readings, such as the confounded norepinephrine severity signal, which I expect to cost well under 0.005 nats. I will save the edit now and confirm that only these three tensors changed. My next forward pass will then compare factual NLL for the original and edited model on the same stays.

```python
W = load_original('embed.numeric.weight').copy()
W[:,2] = 0.0; W[:,4] = 0.0
save_edit('embed.numeric.weight', W)
for n in ['embed.bins.dose.weight','embed.bins.rate.weight']:
    T = load_original(n).copy()
    T[1:17] = T[8]
    save_edit(n, T)
print(edited())
for n in edited():
    a = load(n); b = load_original(n)
    print(n, a.shape, "changed entries:", int((a!=b).sum()), " max abs change", float(np.abs(a-b).max()))
# save the stay list for reuse
json.dump([int(s) for s in sel1], open('/work/sel1.json','w'))
```