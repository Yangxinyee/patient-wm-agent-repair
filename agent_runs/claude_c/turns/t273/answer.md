I'm settling on N5 alone, numeric x5 on the z_dose and z_rate columns. The b_in route is ruled out by the earlier measurement: the dose-induced step along the dose direction is about 4.5 against a natural reading-step sd near 100. N5 is also close to the best numeric factor for the pooled main readout. N20 is worse in every insulin band, and the original is better only for the few high-dose stays. Adding head0 would gain about 0.1 to 0.3 on the readouts for a real risk of failing the factual limit.

I'll undo the 48 head0 ops and check that exactly the two numeric ops remain. Then I'll check N5 on fresh tuning stays.

```python
for _ in range(48): undo()
o=ops(); print(len(o), [(x['name'],x['alpha'],x['side']) for x in o])
print([int(np.argmax(np.abs(np.load('/work/'+x['vector'])))) for x in o])
```