The architecture shows the only way dose information can reach glucose and MAP predictions is through the state update. The rate goes into the action vector u, then into z_post and the prelude, and from there into h and out. The value head reads only the state plus the last value of the same code, so the edit has to put dose-deviation information into the dynamics. The z_post route, using its dead hidden units, looks most promising. Before redoing work I will check what I already built last session: jac2.py, sim.py and the saved JI/JN arrays, which are probably Jacobians of downstream predictions with respect to z. I will also list the core tensor shapes. If JI/JN turn out to be per-lag sensitivities of predictions to z-directions at drug bundles, I can pick an injection direction from them directly.

```python
import numpy as np
for f in ['jac2.py','sim.py']:
    print('=====',f); print(open('/work/'+f).read()[:1700])
for f in ['JI.npy','JN.npy','sIN.npy']:
    a=np.load('/work/'+f); print(f, a.shape, a.dtype, np.round(a.ravel()[:6],3))
cs=[f"{n}{tuple(SHAPES[n])}" for n in NAMES if n.startswith('core.') or n.startswith('z_')]
print(len(cs)); print(' '.join(cs)[:2500])
```