`sim.py` shows how one model step works:
- the flowed hidden state, the context, the z posterior mean;
- then `core_step(hf, e, u, z, M, ages, V)`, where M is the memory of past e tokens and `ages` their ages in seconds;
- weights come from a cached dict `WT`.

So I can add a perturbation to the z token for a single bundle, either by temporarily changing `WT['z_in.bias']` or by calling `core_step` directly. I can then read the next reading's predicted mean with `value_mean`.

Next run:
- check that a single step started from the tool's stored previous state reproduces the stored next state;
- time one call;
- prepare the record lookups (last value and age of each code) for the 100 stays.

If one step takes a few milliseconds, a full sensitivity map over the 192 token directions for about 100 bundles is affordable. A sensitivity map is a regression on the model's states, not training.

```python
import time
trA = pd.read_csv('/data/train.csv')
st100 = np.unique(sid3); trS = trA[trA.stay_id.isin(st100)]
grp = {s: g.sort_values('hour') for s,g in trS.groupby('stay_id')}
dtw = WT['core.flow.a'].dtype
EtW = torch.as_tensor(np.asarray(Eall), dtype=dtw); UtW = torch.as_tensor(np.asarray(Uall), dtype=dtw)
first = np.r_[True, sid3[1:]!=sid3[:-1]]
j0of = np.maximum.accumulate(np.where(first, np.arange(len(sid3)), 0))
def one_step(k, ztok=None):
    o = torch.as_tensor(X3n[k-1,:192], dtype=dtw); h = torch.as_tensor(X3n[k-1,192:960].reshape(4,192), dtype=dtw)
    dtk = dts[k]; hf = h*torch.exp(-dtk*torch.exp(WT['core.flow.a'])); ctx = ctx_from(o, hf, dtk)
    z = zpost_mean(ctx, EtW[k], UtW[k]); j0 = j0of[k]; idx_m = np.arange(max(j0, k+1-128), k+1)
    ages = (tmin[k]-tmin[idx_m])*60.0
    if ztok is not None:
        b_old = WT['z_in.bias'].clone(); WT['z_in.bias'] = b_old + ztok
        try: h1, o1 = core_step(hf, EtW[k], UtW[k], z, EtW[idx_m], ages, {'incl':1,'unit':'s','age':'both'})
        finally: WT['z_in.bias'] = b_old
    else: h1, o1 = core_step(hf, EtW[k], UtW[k], z, EtW[idx_m], ages, {'incl':1,'unit':'s','age':'both'})
    return h1, o1
kk = np.where(hI & ~first)[0][:50]
t0 = time.time(); errs = [np.abs(N(one_step(k)[1]) - X3n[k,:192]).max() for k in kk]; t1 = time.time()
print('per call ms', round((t1-t0)/len(kk)*1000,2), 'out err median', np.median(errs).round(5), 'max', np.max(errs).round(4), 'dts dtype', type(dts), getattr(dts,'dtype',None))
```