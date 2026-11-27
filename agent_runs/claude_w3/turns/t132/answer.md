I re-read the architecture. A dose reaches the state only through the action encoder (u_k), which feeds the prelude and the z posterior. The value head reads `out`, the time-flowed slots and a per-code "last value" register. My notebook says the original model barely stores dose size, which is why its simulated dose effects are near zero.

The repair I'm working towards adds a dedicated, slowly decaying "dose memory" direction to the state. The value head would read that direction, so predicted readings shift by the effect estimated from the records. To build and check it I need my local reimplementation of the forward pass. The kernel has encoder helpers (`enc_bundle`, `atom_emb`, `build_bundles`, ...), but I don't know how complete they are.

Next I'll list the rest of the kernel's functions and the npz contents, keeping the output short. If a full step function exists (prelude, blocks, coda), I'll validate it against the saved states. If only the encoders exist, I'll write the core step next.

```python
import inspect
fns = sorted(k for k, v in globals().items() if callable(v) and not k.startswith('_') and getattr(v, '__module__', None) == '__main__' and inspect.isfunction(v))
print(fns)
rest = sorted(vs.items())[200:]
print([k for k, t in rest if t not in ('function',)])
print(list(np.load('/work/arrays.npz').keys())[:40], list(np.load('/work/theta.npz').keys()))
```