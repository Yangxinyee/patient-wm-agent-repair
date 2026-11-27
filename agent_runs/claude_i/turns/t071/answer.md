The previous output was blocked for being too long, so I'll keep outputs short from now on. My notes stop at fwd_1, but /work also holds fwd_2, fwd_3, enc.py, head.py and F2.pkl, which I haven't written up. First I'll take stock: the kernel variables, the files in each fwd folder, and what enc.py and head.py contain. Then I can pick up without repeating work. Nothing is being tested yet; this is bookkeeping.

```python
import os, json, sys
g = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_') and type(v).__name__ not in ('module','function','type','builtin_function_or_method')}
print(len(g)); print(sorted(g.items())[:80])
for d in sorted(os.listdir('/work')):
    p = os.path.join('/work', d)
    if os.path.isdir(p):
        print(d, os.listdir(p))
    else:
        print(d, os.path.getsize(p))
for f in ['enc.py','head.py']:
    s = open('/work/'+f).read()
    print('-----', f, len(s)); print(s[:1500])
```