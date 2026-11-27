The last output was a short source excerpt showing scaffold, internals, capture_wc and mean_ablate. These are helpers I wrote for the 3-dims-per-drug plan. My notes stop at run 186, so I don't know what runs 187–246 built. Before changing anything I'll take a short inventory: files, function names, and the main variables in the kernel. I'll print names and shapes only, no source listings, to keep the output short. If the inventory shows variables from a partly built write/read pathway, I'll continue that work. If not, I'll rebuild the minimal version from the plan in my notes.

```python
import os, re, json, numpy as np, torch
print(sorted(os.listdir('/work')))
ed='/work/edited'; print('edited dir:', (sorted(os.listdir(ed))[:12] if os.path.isdir(ed) else 'n/a'))
for f in ['lib1.py','lib2.py','lib3.py','lib4.py']:
    src=open('/work/'+f).read(); names=re.findall(r'^def (\w+)\(',src,flags=re.M)
    print(f, len(src.splitlines()),'lines:', ','.join(names)[:300])
g=dict(globals())
fn=sorted(k for k,v in g.items() if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__')
print('funcs',len(fn),','.join(fn)[:1200])
dk=sorted((k,len(v)) for k,v in g.items() if isinstance(v,dict) and not k.startswith('_') and k not in('g',))
print('dicts',str(dk)[:700])
ar=sorted((k,tuple(v.shape)) for k,v in g.items() if isinstance(v,(np.ndarray,torch.Tensor)) and not k.startswith('_'))
print('arrays',len(ar),str(ar)[:1500])
sc=sorted((k,v) for k,v in g.items() if isinstance(v,(int,float)) and not isinstance(v,bool) and not k.startswith('_'))
print('scalars',str(sc)[:600])
print('notes chars',len(open('/work/notes.md').read()))
```