The variable list was cut off at 10,000 characters. Python keeps globals in creation order, so the newest names are at the end. Those are what runs 190–203 created. I will print only the names after `pk`, plus the /work file list, the function names in impl*.py and the notebook headers.

```python
keys=[k for k in globals().keys() if not k.startswith('_')]
i=keys.index('pk')
out=[]
for k in keys[i:]:
    v=globals()[k]
    if isinstance(v, types.ModuleType): continue
    t=type(v).__name__
    if callable(v) and not hasattr(v,'shape'): out.append(f"F:{k}")
    elif hasattr(v,'shape'): out.append(f"A:{k}{tuple(v.shape)}")
    elif isinstance(v,(int,float,str,bool)): out.append(f"S:{k}={str(v)[:25]}")
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
print(len(notes)); print([l[:60] for l in notes.split('\n') if l.startswith('#')])
```