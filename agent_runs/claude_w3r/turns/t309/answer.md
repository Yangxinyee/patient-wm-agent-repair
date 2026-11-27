`wc_row` is fixed to glucose_finger. It reads from `HD`, a head object in the kernel, and accesses an attribute called `wc`, so `HD.wc` is probably a method that takes the code name. I also have 1,694 stored gf head inputs (WC) and the bin centres: gf runs 73–271 mg/dL and map_arterial 32–100 mmHg.

Before fitting gate thresholds and zero-rate baselines, I need to compute head inputs for map rows too, ideally on a grid of last value × age for all four measurement codes so the gate is checked in the worst case. To do that I need the call pattern. Next, I will disassemble `wc_row` into compact instructions and list `HD`'s methods and signatures. If `HD.wc` takes a code argument, the grid check is direct. If it does not, I will assemble the code-specific terms from the embedding tensors.

```python
import dis
ins_ = [f"{i.opname}:{i.argrepr}" for i in dis.get_instructions(wc_row)]
print(' '.join(ins_)[:2200])
print([m for m in dir(HD) if not m.startswith('_')][:60])
for m in [m for m in dir(HD) if not m.startswith('_')]:
    f = getattr(HD, m, None)
    c = getattr(f, '__code__', None) or getattr(getattr(f,'__func__',None),'__code__',None)
    if c is not None: print(m, c.co_varnames[:c.co_argcount])
print(type(GR).__name__, (GR.columns.tolist()[:30] if hasattr(GR,'columns') else ''), getattr(GR,'shape',None))
print(type(zprior).__name__, getattr(zprior,'shape',None))
```