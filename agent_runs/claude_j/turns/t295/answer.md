The helper files are small. gen2(s, conds, start_min=720) in gen2.py looks like the generator that built the cf dataset M. run_cf(s, edits, k_from_min, X) and mod_events_new(evs, edits) do the cf replay, and shifts(s, drug, m, h0) gives true shifts per reading.

To produce exact cf data (doses scaled and readings shifted by the true effect) I need the exact calling convention. This run shows the generator code and which globals the event modifier reads, presumably the SHIFT dict.

```python
print(open('/work/gen2.py').read()); print('-----'); print(open('/work/gen.py').read()[:2006])
print('mod_events_new globals', [x for x in mod_events_new.__code__.co_names][:20]); print('shifts globals',[x for x in shifts.__code__.co_names][:20])
```