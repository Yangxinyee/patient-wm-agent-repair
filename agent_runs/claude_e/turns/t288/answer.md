Original-model NLL is similar on training and tuning batches: training means 1.578–1.658, tuning 1.620–1.643. So there is no strong overfitting signal, and fresh training stays are a reasonable stand-in for unseen stays when measuring the edit's cost.

I drew 600 training stays (6 batches of 100) that appear in none of my selection files or forward passes. The readout-fit rows file failed to load in my exclusion check because of its structure. Before using the new batches, I will make sure none of them were used to fit the insulin readout, since in-sample stays would make the cost look smaller. Any overlapping stay will be replaced with a fresh one.

```python
rr = pd.read_pickle('/work/readrows_tr.pkl'); print(type(rr), (list(rr.keys())[:10] if isinstance(rr, dict) else getattr(rr, 'columns', None)))
ids = set()
def grab(o):
    if isinstance(o, pd.DataFrame):
        for c in ['stay','stay_id']:
            if c in o.columns: ids.update(int(x) for x in o[c].unique())
    elif isinstance(o, dict):
        for v in o.values(): grab(v)
    elif isinstance(o, (list, tuple)):
        for v in o: grab(v)
grab(rr); print('readout-fit stays', len(ids))
batches = json.load(open('/work/sel_cost2.json')); allp = set(x for b in batches for x in b)
ov = allp & ids; print('overlap', len(ov))
if ov:
    pool = [int(x) for x in free if int(x) not in ids and int(x) not in allp]
    rng = np.random.default_rng(7); rep = iter(rng.choice(pool, size=len(ov), replace=False).tolist())
    batches = [sorted(next(rep) if x in ov else x for x in b) for b in batches]
    json.dump(batches, open('/work/sel_cost2.json','w')); print('replaced; overlap now', len(set(x for b in batches for x in b) & ids))
print(json.dumps(batches[0]))
```