Two things since the restart. Runs 190–203 never made it into the notebook. The last result showed that a unit injection reaches the glucose-reading head input (with_code) as one consistent direction: 1,694 rows, mean size 0.0101 per unit, cosine 1.000 to the mean. That supports the plan of an ungated linear channel.

Before going further I need to recover what runs 190–203 built. I will list the kernel's variables and functions, the /work files, the function names in impl*.py, and the notebook's section headers. This step cannot confirm or refute a hypothesis; it only restores my state.

```python
import os, re, types, numpy as np
out=[]
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v, types.ModuleType) or k in ('In','Out','exit','quit','get_ipython'): continue
    t=type(v).__name__
    if callable(v) and not hasattr(v,'shape'): out.append(f"F:{k}")
    elif hasattr(v,'shape'): out.append(f"A:{k}{tuple(v.shape)}")
    elif isinstance(v,(int,float,str,bool)): out.append(f"S:{k}={str(v)[:30]}")
    elif isinstance(v,(dict,list,tuple,set)): out.append(f"C:{k}[{t}{len(v)}]")
    else: out.append(f"O:{k}[{t}]")
print(len(out)); print(' '.join(out))
for f in sorted(os.listdir('/work')):
    p='/work/'+f
    print(f, os.path.getsize(p) if os.path.isfile(p) else sorted(os.listdir(p)))
for f in ['impl.py','impl2.py','impl3.py','impl4.py']:
    src=open('/work/'+f).read()
    print(f, len(src), re.findall(r'^def (\w+)', src, re.M))
notes=open('/work/notes.md').read()
print(len(notes)); print([l[:70] for l in notes.split('\n') if l.startswith('#')])
```