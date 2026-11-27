The factual damage from v4 is confined to the very start of each stay:

| Reading time | Finger glucose bias (edited − original) | MAP bias |
|---|---|---|
| First readings (hour −1) | +51 mg/dL | −23 mmHg |
| Hour 0 | −11 mg/dL | +4 mmHg |
| Hour 1 | −5 mg/dL | 0 |
| Hour 2 | −1 mg/dL | 0 |
| Hour 3 on | about 0 | about 0 |

Once the stay is a few hours in, the edited model matches the original. That holds whether or not drugs ran recently: the NLL change is about 0.000 and the bias about 0.1. So my "drug-free-state" hypothesis is refuted; the leak is specific to the start of the stay.

A change in the prediction for the very first reading means the edit acts before any recorded event is read. There must be some initial processing step, such as a start-of-stay bundle run through the core with the initial slot tokens. Those tokens are tiny (norm under 1), so the normalised input there almost certainly has a large component along my channel direction. `b_in` then writes a big injection into the starting state.

This opens a puzzle. In teacher-forced mode, outside the start, the edit's dose effect is tiny (−0.05 per unit). Yet the fwd_8 rollout showed a +2.3 shift at hour 13 and a steady upward drift. Both rollouts are identical up to hour 12, so this is suspicious. fwd_8 may have been run with a different version of the edit, since the runs in between were not noted.

This run checks three things:
- the edit hash of fwd_8 against the current one (45351ba6ba);
- what ARCHITECTURE.md says about how the initial state is formed;
- the exact times of the hour −1 readings.

If the hashes differ, the fwd_8 rollout says nothing about the current edit. In that case the current edit's main faults are the start-of-stay leak and the weak gain, and I must re-test the rollout. If they match, the free-running rollout really behaves differently from teacher forcing and needs its own investigation.

```python
import json, re
S9 = json.load(open('/work/fwd_9/summary.json'))
print('fwd_8 edit_hash', S8.get('edit_hash'), ' edited tensors', S8.get('edited_tensors'), ' | fwd_9 hash', S9.get('edit_hash', '45351ba6ba (from tool msg)'))
for k in range(5, 9):
    try:
        s = json.load(open(f'/work/fwd_{k}/summary.json')); print(f'fwd_{k}', s.get('type'), s.get('edit_hash'), s.get('edited_tensors'))
    except Exception as ex: print(k, ex)
arch = open('/model/ARCHITECTURE.md').read()
lines = arch.split('\n')
hits = [i for i, l in enumerate(lines) if re.search(r'initial|first bundle|start|static|BOS|begin|init|prior to|before the first', l, re.I)]
for i in hits[:20]: print(i, lines[i][:220])
neg = tun[(tun.hour < 0)]
print('negative-hour rows:', len(neg), neg.event.value_counts().to_dict(), ' hour values', sorted(neg.hour.unique())[:8])
print(f9[f9.h == -1][['stay_id','hour','event','observed','predicted_mean_o','predicted_mean_e']].head(6).to_string())
```