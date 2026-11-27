The notebook ends with a plan to run a dose_edit check of the gated3 edit. Run 137 compared the NE rate-lag coefficients for MAP: the true lag-1 coefficient is about +3.1, the model's is about -0.1. I don't know what fwd_10 was, so first I'll check the start of the notebook, the fwd_10 files and the kernel variables. Then I'll decide on the next forward pass.

```python
import os, json, glob
txt = open('/work/notes.md').read()
print('notes length', len(txt))
head = txt[:max(0, len(txt)-8000)]
print(head[:6000])
print('-----')
for d in sorted(os.listdir('/work')):
    p = '/work/'+d
    if os.path.isdir(p):
        print(d, os.listdir(p))
print('-----')
for f in glob.glob('/work/fwd_10/*'):
    print(f, os.path.getsize(f))
    if f.endswith('.json') or f.endswith('.txt') or f.endswith('.md'):
        print(open(f).read()[:3000])
print('-----')
import types
print([k for k,v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_')])
print('edited', edited())
```