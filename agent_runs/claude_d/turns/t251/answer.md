E4, the E3b insulin edit plus the norepinephrine channel, is saved as the current edit and matches exactly.

Next is a dose_edit "both" run on sel1, the same 100 tuning stays used in forward passes 8 and 10. It will show:
- whether insulin stays near E3b's expected −13.9 and +6.6;
- how much norepinephrine response a lag-1 gain of 2 produces in the rollouts. The targets on these stays are +4.85 and −2.43.

To build the request I need the stay IDs. I will print sel1 and also save a disjoint second tuning subset (sel2) for later validation.

```python
import json
sel1 = json.load(open('/work/sel1.json')); print(len(sel1)); print(json.dumps([int(s) for s in sel1]))
tun = pd.read_csv('/data/tuning.csv', usecols=['stay_id'])['stay_id'].unique()
rest = [int(s) for s in tun if int(s) not in set(int(x) for x in sel1)]
sel2 = [int(s) for s in np.random.default_rng(7).choice(rest, 100, replace=False)]; json.dump(sel2, open('/work/sel2.json','w')); print(len(rest), 'sel2 saved')
```