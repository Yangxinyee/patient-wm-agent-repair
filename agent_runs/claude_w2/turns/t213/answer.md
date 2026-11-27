The last run (207) compared "oracle lag-only kernels" in my local rollout engine with my record-based estimate of the true effect, on 100 train stays.
- **Gain −1.0:** the predicted effect is about 60% of the truth (insulin x2: −9.7 vs −16.5).
- **Gain −1.5 or (−1.2, −0.4):** this gets to about 80% (−13.1 / −13.3 vs −16.5). The shortfall is concentrated at the middle lags (lag bins 2–3: −7 vs −11, −12 vs −17). The late lags are about right.

So a channel that adds the insulin effect into the model's predictions is propagated through the rollout, but not one-for-one. The model's own dynamics partly damp it, especially early.

The conversation was restarted and I only see the last 8,000 characters of my notebook. Before going on I need the earlier part. It holds my record-based "truth" estimates for all four drug/outcome pairs and the exact definition of the oracle edit I was testing. I will print the beginning of notes.md and list the work files. That tells me which scripts (oracle.py, rollout_h2.py, etc.) implement the current edit candidate. This run does not test a hypothesis; it restores context so I do not repeat work.

```python
import os
txt = open('/work/notes.md').read()
print(len(txt))
print(txt[:len(txt)-8000][:14000])
print('-----FILES-----')
for f in sorted(os.listdir('/work')):
    p = '/work/'+f
    print(f, os.path.getsize(p) if os.path.isfile(p) else 'dir')
```