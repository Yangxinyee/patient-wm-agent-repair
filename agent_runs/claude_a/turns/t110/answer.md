The last result showed that my offline copy of the model reproduces the forward tool's teacher-forced value NLL exactly (1.404487 nats per reading on the sel2 training stays). Earlier runs already reproduced the encoder, the core step, the heads and the dose-edit rollout. So I now have a full offline model on which I can test edits before spending forward passes.

**What I currently believe, from the records:**
- **Insulin:** within a glucose tier the rate is randomised. Its total effect on glucose_finger is about −2 per unit/h at lag 1, decaying over about 8 hours (sum about −11). The effect is linear in rate.
- **Norepinephrine:** the effect on MAP is about +3.4 per unit at lag 1, decaying over about 6 hours (sum about +9). It is also linear.
- **The original model:** it barely uses dose magnitude. That is why its readouts are about −1.9 and +1.0 for insulin, and about −0.4 and −0.5 for norepinephrine. My targets on sel1 are −14.3, +7.15, +4.1 and −2.0.

**Next step:** after the restart I only have function objects without their source. Before designing the edit (a dedicated decaying "dose memory" in the state, read out by the glucose and MAP heads), I need an inventory: the functions and their signatures, the stored arrays, and the tensor names and shapes.

**How I will judge it:** this run is bookkeeping, not a test. It succeeds if the rollout, factual and step functions are all present and callable. If they are missing, I will have to rebuild before any edit.

```python
import inspect, types, numpy as np, os, json
g = dict(globals())
funcs, vars_ = [], []
for k,v in g.items():
    if k.startswith('_'): continue
    if isinstance(v, types.FunctionType):
        try: sig = str(inspect.signature(v))
        except Exception: sig='(?)'
        funcs.append(f"{k}{sig}")
    elif isinstance(v,(types.ModuleType, type)): continue
    elif isinstance(v, np.ndarray): vars_.append(f"{k}:nd{v.shape}")
    elif hasattr(v,'shape') and hasattr(v,'dtype'): vars_.append(f"{k}:T{tuple(v.shape)}")
    elif isinstance(v,(dict,list,tuple,set)): vars_.append(f"{k}:{type(v).__name__}[{len(v)}]")
    elif isinstance(v,(int,float,str,bool)): vars_.append(f"{k}={str(v)[:30]}")
    else: vars_.append(f"{k}:{type(v).__name__}")
print("FUNCS:\n" + "\n".join(funcs))
print("\nVARS:", "; ".join(vars_))
print("\n/work/edited:", os.listdir('/work/edited')[:20], " fwd_4:", os.listdir('/work/fwd_4'))
dc = np.load('/work/dimcost.npy'); print('dimcost', dc.shape, 'lowest', np.round(np.sort(dc.ravel())[:12],5))
print("\nTENSORS:", " ".join(f"{n}:{tuple(SHAPES[n])}" for n in NAMES))
```