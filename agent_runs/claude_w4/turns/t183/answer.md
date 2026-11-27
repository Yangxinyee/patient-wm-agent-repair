Run 156 showed the architecture. Dose information enters only through the action encoder: it reaches the prelude as the u_k token, which heads 1–3 read at drug bundles. Glucose predictions are read from `out` and the slots. The earlier listing was too long, so I'll keep outputs short. Next I'll list only the functions I defined and the shapes of a few key variables from runs 144–155. That tells me which replica tools (step_full and others) and which surprise directions (nuG, nuM) I already have before designing the prelude-level injection.

```python
import types
fns = sorted([k for k,v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_')])
print('functions:', ', '.join(fns))
for nm in ['nuG','nuM','D4','yt','ot','CTX3','OUT3','H3','Xall','UI_all','UN_all','fI','fN','units4']:
    v = globals().get(nm, None)
    if v is None: print(nm, 'missing'); continue
    sh = getattr(v,'shape',None)
    print(nm, type(v).__name__, sh if sh is not None else (len(v) if hasattr(v,'__len__') else ''))
```