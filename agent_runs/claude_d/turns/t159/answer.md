The last output was a set of linear-separability checks on the value-head input vectors (`with_code`):
- Glucose rows and MAP rows can be told apart perfectly by a linear readout (r2 = 1.0, clean margin).
- Among glucose rows, "insulin in the previous bundle" is nearly linearly decodable (r2 0.996, small overlap).
- The combined gate "glucose AND insulin", fitted over all rows, is not linearly separable (r2 0.78). So a gated correction must factor into a code gate times a drug-signal term, not one linear direction.
- The NE gate among MAP rows works for almost all rows (q01 0.82), with a few outliers.

My notebook stops at run 129 and does not mention forward passes 5 to 7 or `selT.json`. Before I build the edit, I need to recover what the kernel holds: the helper functions, the arrays from those passes, and any decoders already fitted. This run only inspects state and records it. It confirms the plan if I find the head-reproduction helpers and the states/factual data from fwd_5 to fwd_7. If those are missing, I will have to rebuild them.

```python
import types, os, json, numpy as np
out=[]
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v, types.ModuleType) or k in ('In','Out','exit','quit','get_ipython'): continue
    if isinstance(v, np.ndarray): out.append(f"A:{k}{v.shape}")
    elif hasattr(v,'shape') and not callable(v): out.append(f"T:{k}{tuple(v.shape)}")
    elif callable(v): out.append(f"F:{k}")
    elif isinstance(v,(int,float,np.floating)): out.append(f"S:{k}={float(v):.4g}")
    elif isinstance(v,str): out.append(f"s:{k}={v[:30]!r}")
    elif isinstance(v,(list,tuple,dict,set)): out.append(f"C:{k}[{type(v).__name__}{len(v)}]")
    else: out.append(f"O:{k}:{type(v).__name__}")
print(len(out)); print("; ".join(out))
for d in sorted(os.listdir('/work')):
    p=os.path.join('/work',d)
    if os.path.isdir(p):
        print(d, [(f, os.path.getsize(os.path.join(p,f))) for f in sorted(os.listdir(p))])
for f in ['sel1.json','selS.json','selT.json']:
    s=json.load(open('/work/'+f)); print(f, type(s).__name__, len(s), str(s)[:80])
```