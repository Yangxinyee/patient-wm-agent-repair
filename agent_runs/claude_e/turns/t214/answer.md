The true insulin effect is linear in dose:
- slope on positive deviations: −2.08 (se 0.07);
- slope on negative deviations: −1.99 (se 0.10);
- squared term: −0.055 (se 0.03), negligible.

So the true ×2/×½ ratio is −2, and the model's 2.4 is a model artefact. Insulin ×½ could be lifted with a steeper slope for negative readouts, but that would add roughly 0.001 per reading of factual cost. Whether I can afford it depends on the factual cost of the current a = −2.6 edit on random stays. The offline sel1 estimate is 0.0037.

I'll measure the cost properly with the tool on tuning stays. I'll get the full list of 400 tuning stay ids, check whether they are consecutive, and write the three batches other than f3 to files so the requests are exact.

```python
tu_ids = sorted(map(int, stays_tu))
print(len(tu_ids), tu_ids[:3], tu_ids[-3:], 'consecutive:', tu_ids == list(range(tu_ids[0], tu_ids[0]+len(tu_ids))))
rest_ = [s for s in tu_ids if s not in set(map(int, f3_ids))]
batches_tu = [rest_[i*100:(i+1)*100] for i in range(3)]
for i, b in enumerate(batches_tu):
    json.dump(b, open(f'/work/sel_tu{i}.json','w'))
    print(i, len(b), b[0], b[-1], 'consecutive', b == list(range(b[0], b[0]+len(b))))
with open('/work/notes.md','a') as fh:
    fh.write("\n- fwd_12: dose_edit both on f3 (2020000000..99) with ins_only(-2.6) [hash 924a84db1e]: orig x2 -1.85, half +.78; edited x2 -13.01 (truth -13.18 -> ratio .98), half +5.41 (truth 6.59, ratio .80); NE unchanged (-.51/-.62 vs truth 4.76/-2.38). (fwd_10 was on sel10, a different set!)\n- Run 198: TRUE insulin effect linear: dev1 pos slope -2.08 (.07), neg -1.99 (.10), quad -.055 (.03). Truth x2/half ratio = -2; model asymmetry 2.4 is artifact.\n- Next: factual both at -2.6 on all 400 tuning stays (sel_f3 + /work/sel_tu0..2.json).\n")
```