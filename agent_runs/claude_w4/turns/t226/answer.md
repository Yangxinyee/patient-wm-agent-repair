The kernel is intact: the readout dicts (BB3 with tI/fN/pI, RO, TT), the head-1 directions aI3, aN3, aP3 (48-dim each), the replica tensors and the cached action tokens are all there. The listing was cut again before the `build_v3` internals, which are what matter.

This time I'll print only `build_v3`'s global names, locals and constants, its disassembly in short form, and the shapes and norms of the arrays it relies on (dI, dN, qd and anything else it references). That tells me exactly which tensors v3 would edit and whether its inputs are still consistent. If it only touches `core.prelude.*` attention kv/out weights for head 1, I'll build it at a modest gain and verify it numerically on the CPU replica before spending a forward pass.

```python
import types, dis, io, contextlib
c=build_v3.__code__
print('names:', c.co_names)
print('locals:', c.co_varnames)
print('consts:', [x for x in c.co_consts if not isinstance(x,types.CodeType)])
buf=io.StringIO()
with contextlib.redirect_stdout(buf): dis.dis(build_v3)
txt=buf.getvalue().splitlines()
keep=[]
for l in txt:
    p=l.split()
    if not p: continue
    for op in ('LOAD_GLOBAL','LOAD_CONST','STORE_FAST','LOAD_ATTR','CALL','BINARY_OP','BINARY_SUBSCR','STORE_SUBSCR','LOAD_METHOD','RETURN_VALUE','BUILD_SLICE'):
        if op in l:
            keep.append(l.strip()[-60:]); break
print(len(txt),'dis lines')
print(' ; '.join(k.split(None,1)[-1] if len(k.split())>1 else k for k in keep)[:5000])
for nm in c.co_names:
    if nm in globals():
        v=globals()[nm]
        if isinstance(v,np.ndarray): print(nm, v.shape, 'norm %.4g'%np.linalg.norm(v))
        elif isinstance(v,dict): print(nm,'dict',{k:(getattr(vv,'shape',None) if hasattr(vv,'shape') else type(vv).__name__) for k,vv in list(v.items())[:6]})
        elif isinstance(v,(int,float)): print(nm,v)
```