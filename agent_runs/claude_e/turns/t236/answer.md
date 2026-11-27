The source of those functions isn't retrievable (they were defined in earlier notebook runs). I will use their signatures, local variable names and constants as a guide. I will also search my full notebook for what I recorded about the encoder and z posterior checks, so I know how closely they matched the tool. That tells me whether I can build the rest of the core on them or have to start over.

```python
for fn in ['stay_bundles','split_atoms','atom_vec','set_encode','zpost_check','zpost_check2','ctx_from_state','act_fn','mlp','z_draws','q_from_rows']:
    f_=globals()[fn]
    print(fn, inspect.signature(f_), f_.__code__.co_varnames[:f_.__code__.co_nlocals][:25], [c for c in f_.__code__.co_consts if isinstance(c,(str,float,int)) and c not in (None,)][:20])
notes=open('/work/notes.md').read()
import re
for kw in ['zpost','z post','encoder','atom_vec','set_enc','prelude','simulat']:
    for mt in re.finditer(kw,notes):
        st=max(0,mt.start()-200); print('...',notes[st:mt.start()+300].replace('\n',' | ')); print('====')
        break
```