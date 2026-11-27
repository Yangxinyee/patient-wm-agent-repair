v3's edit barely reaches the state. At insulin bundles it moves `out` by a median of 0.004 and the mean slot by 0.2–0.3. That is about 0.2 mg/dL-equivalents, roughly 10–50 times too small, and in an unknown direction.

The reason is in the step structure. The head-1 vector enters the slot tokens, whose norm is about 795. `prelude_norm` then divides by σ ≈ 57. After that, E reaches h only through `softplus(delta_in)*b_in(E)`, plus small effects through the blocks, where E is a small part of h+E. Scaling v3 up would distort the prelude normalisation everywhere.

Better plan: a dedicated channel.
- Head 1 writes c·rate along a slot-token direction q, chosen where the natural tokens hardly vary.
- A rank-1 addition to `core.b_in` reads q from E and writes a chosen target into the new state. The target is the model's own glucose-surprise direction (or MAP for NE), with an arbitrary gain.
- Centering by expected rate comes later.

To design this I need the core tensors, their sizes and gains: `b_in`, `delta_in`, `a_in`, `c_out`, coda, norms and slots. I also need the actual h-slot norms. If `softplus(delta_in)` and `b_in` show a usable linear path from E into h, the channel is feasible.

```python
core = [n for n in NAMES if n.startswith('core.') or n.startswith('encode_action') ]
for n in core: print(n, SHAPES[n])
print('other heads:', [n for n in NAMES if not n.startswith(('core.','encode','embed'))][:60])
sp = lambda x: np.log1p(np.exp(x))
di = np.asarray(load_original('core.delta_in')).ravel(); ai = np.asarray(load_original('core.a_in')).ravel()
print('softplus(delta_in) q', np.round(np.quantile(sp(di),[0,.1,.5,.9,1]),4), 'a_bar q', np.round(np.quantile(np.exp(-sp(di)*np.exp(ai)),[0,.1,.5,.9,1]),4))
H = xo[:,192:960].reshape(-1,4,192); print('h slot norm q', np.round(np.quantile(np.linalg.norm(H,axis=2),[.01,.5,.99]),1), 'out norm', np.round(np.quantile(np.linalg.norm(xo[:,:192],axis=1),[.01,.5,.99]),2))
sl = np.asarray(load_original('core.slots')); print('core.slots', sl.shape, np.round(np.linalg.norm(sl.reshape(-1,192),axis=1),1))
print('mean_h std per dim q', np.round(np.quantile(H.mean(1).std(0),[0,.1,.5,.9,1]),3))
```