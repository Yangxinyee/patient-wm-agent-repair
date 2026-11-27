```python
print(open('/work/enc.py').read())
print("=====")
print(open('/work/head.py').read())
print("=====")
fn=[k for k,v in globals().items() if not k.startswith('_') and type(v).__name__=='function']
print(fn)
print("=====")
for nme in NAMES:
    if nme.startswith('core') or nme.startswith('z_') or nme.startswith('encode_action') :
        print(nme, SHAPES[nme])
```