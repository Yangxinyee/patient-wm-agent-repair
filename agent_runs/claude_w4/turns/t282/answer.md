The architecture shows why v3 can fail even though the carrier is clean. Head 1 adds a vector proportional to the insulin rate to all four slot tokens. That vector then passes through the prelude FFN, `prelude_norm`, the recurrent block and the coda. Each of these normalises or mixes it, so the chosen direction `dI` may not move the glucose belief the intended way.

The rollout helps here. It feeds the model's own sampled readings back into the register and memory. So a dose-driven shift in the next reading's prediction gets carried forward by the persistence I measured (about 0.96 per hour). I only need a slot injection that reliably moves the next glucose prediction by the right amount.

The next step is to measure what v3 actually did, using fwd_7 (original vs edited states on the same factual stays). I'll take Δout at insulin bundles and the following bundles and project it onto a ridge readout of latest glucose fitted on original states. A negative projection, roughly proportional to rate, would mean the direction is right and fwd_6 failed for another reason. A positive or rate-independent projection would mean the direction is wrong. First I need the record format and must check that fwd_7 matches fwd_3, since that code was lost.

```python
print(open('/data/README.md').read()[:4000])
tun = pd.read_csv('/data/tuning.csv'); print(tun.shape); print(tun.dtypes); print(tun.head(8).to_string())
s7o = np.load('/work/fwd_7/states_original.npz'); s7e = np.load('/work/fwd_7/states_edited.npz'); s3 = np.load('/work/fwd_3/states_original.npz')
print({k:s3[k].shape for k in s3.files})
print('same rows fwd3/fwd7:', s3['x'].shape==s7o['x'].shape and np.array_equal(s3['stay_id'],s7o['stay_id']), float(np.abs(s3['x']-s7o['x']).max()) if s3['x'].shape==s7o['x'].shape else None)
print('stays in sel100:', set(np.unique(s7o['stay_id']).tolist())==set(sel))
print('hour sample', s7o['hour'][:10], s7o['has_insulin'][:10], s7o['n_atoms'][:10])
V = VOCAB; print(V.keys()); print(V.get('config'))
for k in ['value_bin_centres','value_bin_edges']:
    if k in V: print(k, {c:(np.round(v,1).tolist() if isinstance(v,list) else v) for c,v in list(V[k].items())[:8]} if isinstance(V[k],dict) else V[k])
```