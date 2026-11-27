The last run failed with a shape mismatch: I multiplied a 32-wide z vector by a 192-row matrix. That was a bug in my code and says nothing about the model. My context was just reset, so before I design the weight edit I need to re-orient. I will check the structure of ARCHITECTURE.md (length and section headers), the helper functions still in the kernel, and the files in /work. This run is bookkeeping, so there is no hypothesis to test. The output should show which sections cover the state update (h slots, decay) and the value head, so I can read only those next.

```python
import os
A = open('/model/ARCHITECTURE.md').read()
print('ARCH chars', len(A))
for i,l in enumerate(A.split('\n')):
    if l.startswith('#'): print(i, l[:90])
fs = sorted(k for k,v in list(globals().items()) if callable(v) and not k.startswith('_') and getattr(v,'__module__',None) in (None,'__main__'))
print('funcs', len(fs)); print(' '.join(fs)[:1500])
for f in sorted(os.listdir('/work')):
    p='/work/'+f
    print(f, os.path.getsize(p) if os.path.isfile(p) else 'dir:'+','.join(os.listdir(p))[:120])
```