Fwd_26 used tuning batch tu0 (stays 2020000100–199). On that batch the saved insulin-only edit costs +0.00459 nats per reading. That matches the prediction from the two earlier settings (.00428 at -2.6, .00492 at -2.8). tu0 is the most expensive batch seen so far. Averaged over all 8 batches the cost is about .003, so the current edit sits inside the .005 budget with some margin.

The architecture summary shows the value head reads only `out` and the slot mean of h. This confirms the earlier finding: the head has only a weak view of the norepinephrine rate level, so any NE readout from the head is noisy.

One option I dropped earlier may still be worth re-checking. At a_ne=3, the head-level NE readout improved the NE x1/2 readout a lot (-0.70 → -1.91), cost about .0013 per reading on random stays, and made x2 slightly worse. A smaller NE gain combined with a slightly lower insulin gain might give a better overall result within budget.

Before spending forward passes on this, I will recover how the NE part of `gated3` was built and what offline cost tools exist. If the combined cost on tu0 cannot stay below about .0047, I will keep the current edit.

```python
import inspect, re
cal = [k for k,v in globals().items() if callable(v) and not k.startswith('_')]
print(cal[80:])
notes = open('/work/notes.md').read()
hits = [l for l in notes.splitlines() if re.search(r'a_ne|NE gate|131|159|gated3|NE part|ne_part|lp_ne', l)]
print(len(notes), len(hits)); print('\n'.join(h[:220] for h in hits[:40]))
for fn in ['gated3','ins_only']:
    if fn in globals():
        s = inspect.getsource(globals()[fn]); print('----', fn, len(s)); print(s[:2500])
```