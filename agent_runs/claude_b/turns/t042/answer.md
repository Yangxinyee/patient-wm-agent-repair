Where things stand:

- **The edit works mechanically.** In fwd_3, the magnitude-blind model gives shifts of exactly 0.0 (shift_se 0) on all four readouts.
- **Factual fit is unharmed.** In fwd_2 on 100 tuning stays, the edited model's NLL was 1.33348 against 1.33354 for the original.

So the edit reflects my belief that neither drug has any effect. The main risk left is that this belief is wrong, in particular that a small per-dose effect builds up over 12 hours of a sustained x2 change.

Next run:

- **Spread of the NLL change.** I'll measure the per-reading spread of the NLL difference from fwd_2. That tells me how safe the 0.005 margin is on unseen stays.
- **What is already loaded.** I'll list the kernel's variables so I can reuse the data I loaded earlier.

After that, I'll run one consolidated test of the sustained (cumulative, 12-hour) effect. The null would be refuted if the summed lag effects are clearly non-zero.

```python
f2o = pd.read_csv('/work/fwd_2/factual_original.csv'); f2e = pd.read_csv('/work/fwd_2/factual_edited.csv')
d = f2e['nll'].values - f2o['nll'].values
print('per-reading nll diff: mean %.6f sd %.5f se %.6f max|d| %.4f' % (d.mean(), d.std(), d.std()/np.sqrt(len(d)), np.abs(d).max()))
print(f2e.assign(d=d).groupby('event')['d'].agg(['mean','std','count']))
g = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_')}
dfs = {k: (globals()[k].shape if hasattr(globals()[k],'shape') else None) for k,t in g.items() if t in ('DataFrame','ndarray','Series','dict','list')}
print({k:v for k,v in dfs.items()})
print([k for k,t in g.items() if t=='function'])
print(open('/data/README.md').read()[:3000])
```