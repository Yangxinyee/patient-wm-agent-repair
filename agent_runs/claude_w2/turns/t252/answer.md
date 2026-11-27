The restart left a gap: my notebook stops at run 207, but the last output is from run 244, a comparison of decoder variants ("base", "wcm", "wc" at ridge 0.01/0.1). It reports decoding R², a factual error and a 7-number "cf" response list. I can't interpret those numbers without my code. Before doing anything new, I will rebuild my context. I'll read the full notebook, list the /work files by modification time to see what I built after run 207, and list the kernel's variables. Nothing is being tested yet; this run only gathers state.

```python
import os, time, types
txt=open('/work/notes.md').read()
print("NOTES LEN", len(txt))
print(txt[:max(0,len(txt)-8000)])
print("=====FILES by mtime")
for f in sorted(os.listdir('/work'), key=lambda f: os.path.getmtime('/work/'+f)):
    p='/work/'+f
    print(f, os.path.getsize(p), time.ctime(os.path.getmtime(p)))
print("=====GLOBALS")
gl=[(k,type(v).__name__) for k,v in list(globals().items()) if not k.startswith('_') and not isinstance(v,types.ModuleType)]
print(len(gl))
print(', '.join(f"{k}:{t}" for k,t in gl))
```