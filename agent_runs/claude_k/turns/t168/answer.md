There are more kernel objects than the listing showed, and it was cut again. The ones that matter are there:
- `drug_atoms(i, t, end_mode, mult)` builds the drug atoms of a bundle.
- `setenc(V, pre)` is my set-encoder rebuild.
- `E_EMPTY` is the measurement encoding of an empty set. So I already learned once that empty sets are not encoded as zero for e, and probably missed the same for u.

Now I'll test that directly. I'll compute the action encoder on an empty set and check its projection on the two decoders. If p_ins·u_empty and p_ne·u_empty are around −9 or lower, that explains the constant −1.5 and −1.0 offsets. I'll also print the rest of the object list.

```python
names=list(globals().keys()); i0=names.index('z_action_err')+1
lst=[]
for k in names[i0:]:
    v=globals()[k]
    if k.startswith('_') or isinstance(v,types.ModuleType): continue
    if callable(v) and not hasattr(v,'shape'):
        try: lst.append(k+str(inspect.signature(v)))
        except Exception: lst.append(k+'()')
    elif hasattr(v,'shape'): lst.append(k+str(tuple(v.shape)))
    elif isinstance(v,(list,dict,tuple)): lst.append(k+':'+type(v).__name__+str(len(v)))
    else: lst.append(k+'='+repr(v)[:20])
print(' | '.join(lst)[:3000])
p_ins_=np.load('/work/p_ins.npy'); p_ne_=np.load('/work/p_ne.npy'); print('p norms',np.linalg.norm(p_ins_),np.linalg.norm(p_ne_))
for pre_ in ['encode_action','encode_action.']:
    try:
        U0=np.asarray(setenc([],pre_)); print(pre_,'U_EMPTY norm',np.linalg.norm(U0),'p_ins.U0',float(p_ins_@U0),'p_ne.U0',float(p_ne_@U0)); break
    except Exception as ex: print(pre_,'err',repr(ex)[:200])
print('E_EMPTY norm',np.linalg.norm(E_EMPTY))
```