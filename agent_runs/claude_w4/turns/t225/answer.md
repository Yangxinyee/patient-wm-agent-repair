The inventory confirms that the kernel still has the full toolkit:
- replica pieces: `prelude`, `mha`, `step_full`, `ffn`;
- the action-token encoders: `u_for`, `u_from_atoms`, `tok_with`;
- the truth calculators: `truth`, `truth_on`, `true_shift`;
- the edit builders: `build_v3` and `build_edit`;
- the fitted ridge readouts `fI`, `fN`, `EI` and `EN`.

The output was cut before the arrays, dicts and scalars, and before the internals of `build_v3`. Those are what I need to judge whether v3 is ready to test.

Next I'll print only that remaining part, compactly. I'll skip DataFrames, sklearn models and functions, and list the arrays with their shapes and the dict keys. I'll also list the global names and constants `build_v3` uses and the disassembled call structure of `build_v3`, so I can see which tensors it edits and how gI, gN and sP enter.
- **Confirms v3 is usable:** it edits only the head-1 value rows (kv rows 240:288) and the output columns 48:96 of one prelude attention.
- **Calls for a rebuild:** it depends on variables that no longer exist.

```python
import types, dis, io, contextlib
skip=(types.ModuleType, types.FunctionType, type)
out=[]
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v, skip): continue
    tn=type(v).__name__
    if tn in ('DataFrame','DataFrameGroupBy','Ridge','LinearSVC','KFold','Generator','ABCMeta','Counter','NpzFile','NoneType','HistGradientBoostingRegressor','Series','GroupKFold'): continue
    if isinstance(v,np.ndarray): out.append(f'nd {k}:{v.shape}')
    elif isinstance(v,torch.Tensor): out.append(f'pt {k}:{tuple(v.shape)}')
    elif isinstance(v,dict): out.append(f'dict {k}:{len(v)}:{[str(x)[:14] for x in list(v.keys())[:6]]}')
    elif isinstance(v,(list,tuple)): out.append(f'{tn} {k}:{len(v)}')
    elif isinstance(v,(int,float,bool,np.floating,np.integer)): out.append(f'sc {k}={v:.4g}' if isinstance(v,(float,np.floating)) else f'sc {k}={v}')
    elif isinstance(v,str): out.append(f'str {k}={v[:30]}')
    else: out.append(f'{tn} {k}')
print(' | '.join(sorted(out)))
c=build_v3.__code__
print('\nbuild_v3 names:', c.co_names)
print('locals:', c.co_varnames)
print('consts:', [x for x in c.co_consts if not isinstance(x,types.CodeType)])
buf=io.StringIO()
with contextlib.redirect_stdout(buf): dis.dis(build_v3)
lines=[l for l in buf.getvalue().splitlines() if ('CALL' in l or 'LOAD_GLOBAL' in l or 'LOAD_CONST' in l or 'STORE_FAST' in l or 'BINARY' in l or 'LOAD_ATTR' in l)]
print('\n'.join(l.split(')')[0][-60:] if False else l[-70:] for l in lines[:160]))
```