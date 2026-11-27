**What the last result showed.** Run 91 checked whether one fixed direction in the value head's input space can carry a dose-driven shift of the predicted mean.

- **glucose_finger:** the direction that moves each reading's predicted mean is fairly consistent across readings (median cosine 0.73). The size of the response varies a lot, though: the response sd is 1.58 for a mean response of 1.0.
- **map_arterial:** the per-reading directions barely agree (median cosine 0.29), and the response sd is 4.6.
- **Decoding:** reading a dose signal back out of the current state with a linear readout is noisy (residual sd about 1.6 rate units).

**What I conclude.** Adding a fixed vector to the state or view will not give a clean, calibrated dose effect. That holds especially for MAP. A workable edit probably needs two things:

1. A dedicated channel that stores past infusion rates and decays over the right time span.
2. A readout that shifts the predicted value distribution in the bin-logit space, not the view space.

**What I will do next and why.**

- **Recover context.** The restart lost my detailed memory of the update equations, so I will re-read ARCHITECTURE.md. I will also list the helper functions still in the kernel (offline teacher-forced and rollout replicas) and the tensor shapes.
- **Find the write path.** I am looking for a path where the infusion rate can be written roughly linearly into some state dimensions with a chosen decay. I will check the action encoder, the h update (`abar`, `sp`, `b_in`) and the `flow.a` decay.
- **Find the read path.** I am also looking for value-head units that can be turned into a near-linear readout.

**What would confirm or refute this plan.**

- **Confirm:** the architecture has a near-linear write path from the action encoding into h, plus spare (untrained or low-use) dimensions or units I can repurpose.
- **Refute:** every path from the dose to the head passes through normalisation with a state-dependent gain I cannot control. In that case I would fall back to calibrating a logit-space edit empirically with the offline rollout replica.

```python
import inspect, types
fn = sorted(k for k,v in list(globals().items()) if isinstance(v, types.FunctionType) and not k.startswith('_'))
sigs=[]
for k in fn:
    try: sig = str(inspect.signature(globals()[k]))
    except Exception: sig = '?'
    sigs.append(k+sig)
print('FUNCS:', '; '.join(sigs))
out=[]
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v,(types.FunctionType, types.ModuleType, type)) or callable(v): continue
    s = getattr(v,'shape',None)
    if s is None:
        try: s = len(v)
        except Exception: s=''
    out.append(f'{k}:{type(v).__name__}{tuple(s) if hasattr(s,"__iter__") else s}')
print('VARS:', ' '.join(out))
print('TENSORS:', ' '.join(f'{n}{tuple(SHAPES[n])}' for n in NAMES))
print('---ARCH')
print(open('/model/ARCHITECTURE.md').read())
```