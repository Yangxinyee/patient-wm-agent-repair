The file /work/sel1.json is a dict with a single key, not a list, so the conversion failed. I will read the list from inside it, print it, and then create sel2.

```python
d_ = json.load(open('/work/sel1.json')); print(type(d_).__name__, list(d_.keys())[:5] if isinstance(d_, dict) else '')
sel1 = d_[list(d_.keys())[0]] if isinstance(d_, dict) else d_
print(len(sel1)); print(json.dumps([int(s) for s in sel1]))
tun = pd.read_csv('/data/tuning.csv', usecols=['stay_id'])['stay_id'].unique()
s1 = set(int(x) for x in sel1); rest = [int(s) for s in tun if int(s) not in s1]
sel2 = [int(s) for s in np.random.default_rng(7).choice(rest, 100, replace=False)]; json.dump(sel2, open('/work/sel2.json','w')); print(len(rest), 'sel2 saved')
```