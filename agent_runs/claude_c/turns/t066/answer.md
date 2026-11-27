The notebook stops at pass 5, so the last session did not record what the 30 `embed.bins` rank1 ops are or what passes 6 and 7 found. The printout of the ops list hit the output limit before I saw the kernel variables. I will now print the ops compactly, including what each vector is, list the globals, and read the outputs of passes 6 and 7. If those passes show the bins ops were a failed or unfinished experiment, I will clear them.

```python
import numpy as np, glob
o=ops()
for i,op in enumerate(o):
    v=np.load('/work/'+op['vector'])
    nz=np.nonzero(np.abs(v)>1e-6)[0]
    print(i+1, op['name'].split('.')[-2], op['alpha'], op['side'], v.shape, 'nz idx', nz[:6], np.round(v[nz[:6]],3))
print("=====GLOBALS=====")
g=[k for k in globals().keys() if not k.startswith('_') and k not in ('In','Out','exit','quit','get_ipython')]
print(g)
for k in [5,6,7]:
    d=f'/work/fwd_{k}'
    print('==',d, os.listdir(d))
    for f in os.listdir(d):
        if f.endswith('.json') or f.endswith('.txt') or f.endswith('.md'):
            print(f, open(os.path.join(d,f)).read()[:1500])
```