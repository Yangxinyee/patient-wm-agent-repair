The architecture text makes the dose pathways explicit. A dose reaches the glucose/MAP heads only through the recurrent state (h slots and out, read through the view and q). There are two routes into that state: the action token u_k in the prelude attention, and the z posterior mean via the z_in token. Every stage passes through LayerNorms and is mixed by c_out and the coda at each bundle.

Forward pass 9 showed writer v2 left the factual fit unchanged: glucose_finger NLL 1.5124 vs 1.5123, overall 1.5202 vs 1.5201. Together with the flat dose-edit result, either the writes never reach the state, or the prelude/LayerNorm path squashes them.

Next I will use the existing states "both" pass (fwd 10) to see where the signal dies:
- Is the z posterior actually changed at drug bundles?
- How large are the downstream differences in out and h, compared with natural variation?
- Which core parameters (a_bar, time-flow decay, c_out, coda) could support a persistent dose memory?

Possible outcomes:
- z changes but out/h barely move: the z-token route is dead, and I need a different channel.
- z doesn't change: the gate never fired.

```python
import numpy as np
T = lambda x: x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
cn = [n for n in NAMES if n.startswith(('core.', 'z_in', 'z_head', 'z_prior'))]
print([(n, tuple(SHAPES[n])) for n in cn])
zi = T(load('z_in.weight')); zo = T(load_original('z_in.weight'))
print('z_in col norms orig/edit:', [(c, round(float(np.linalg.norm(zo[:, c])), 3), round(float(np.linalg.norm(zi[:, c])), 1)) for c in [17, 18, 21, 26, 27, 28]])
A = np.load('/work/fwd_10/states_original.npz'); Bz = np.load('/work/fwd_10/states_edited.npz')
xo = A['x']; xe = Bz['x']; dx = xe - xo
hi = A['has_insulin'].astype(bool); hn = A['has_norepinephrine'].astype(bool); nd = ~(hi | hn)
print('rows', len(xo), 'ins', hi.sum(), 'ne', hn.sum())
for nm, sl in [('out', slice(0, 192)), ('h0', slice(192, 384)), ('h1', slice(384, 576)), ('h2', slice(576, 768)), ('h3', slice(768, 960)), ('z', slice(960, 992))]:
    print(nm, 'diff rms drug %.4f nondrug %.4f | natural sd %.3f' % (np.sqrt((dx[~nd, sl]**2).mean()), np.sqrt((dx[nd, sl]**2).mean()), xo[:, sl].std(0).mean()))
zd = dx[:, 960:992]
print('z diff rms/dim ins bundles', np.round(np.sqrt((zd[hi]**2).mean(0)), 3))
print('z diff rms/dim non-drug  ', np.round(np.sqrt((zd[nd]**2).mean(0)), 3))
sp = lambda x: np.log1p(np.exp(x))
di = T(load_original('core.delta_in')); ai = T(load_original('core.a_in'))
ab = np.exp(-sp(di) * np.exp(ai)); print('a_bar', ab.shape, np.quantile(ab, [0, .1, .5, .9, 1]).round(3), 'n>.9', (ab > .9).sum())
fa = T(load_original('core.flow.a')); print('flow tau hours quantiles', fa.shape, np.quantile(1 / np.exp(fa) / 3600, [0, .1, .5, .9, 1]).round(3))
```