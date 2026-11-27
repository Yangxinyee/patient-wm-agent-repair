The tensor inventory and config are complete enough to rebuild the core:
- d = 192, 4 heads, 4 slots, memory of 128;
- a prelude layer, two recurrent layers, one coda layer;
- set encoders and the z prior/posterior MLPs.

Some details are undocumented, such as the 8 inputs to `last_numeric`, `embed.kq` and the memory age units. I can settle them by checking against the tool's saved states: out, the four h slots and the z posterior mean after each bundle.

This is worth doing because NE x2 is mostly an out-of-range problem. Without the model's state under edited NE doses, I cannot design a readout that responds to them. Recorded NE never exceeds 1.0, so a term that fires only on the out-of-range state movement should cost almost nothing factually.

I already have a local value head that matches the tool, so parts of the view computation exist. Before writing anything new, I will list what the kernel and /work/lib.py already define, so I can reuse it rather than rebuild it.

```python
import inspect, types
fns=[k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_')]
print('functions:',fns)
big=[(k,type(v).__name__,getattr(v,'shape',None) or (len(v) if hasattr(v,'__len__') else '')) for k,v in globals().items()
     if not k.startswith('_') and isinstance(v,(np.ndarray,list,dict,pd.DataFrame)) and k not in ('NAMES','SHAPES','VOCAB')]
print(big[:80])
src=open('/work/lib.py').read(); print(len(src)); print([l for l in src.splitlines() if l.startswith('def ')])
```