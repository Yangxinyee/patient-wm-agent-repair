The last run listed the helper functions and the stored read directions (DQ, RD, GD, DECL). The rate-carrying signal is in place up to the core state, but no tensors are saved yet. The stage still missing is the value head: it must turn that signal into a shift of the predicted glucose_finger and MAP means, and only for those two codes.

Next, I will read the value-head section of ARCHITECTURE.md (coarse and fine bins, embed_transition) and check which build variables are still in the kernel. I will print summaries only. If the value head is a two-layer network over 17 coarse bins with inactive hidden units, I will use two of those units per drug to tilt the logits. If not, I will have to choose a different route.

```python
import re
A = open('/model/ARCHITECTURE.md').read()
paras = A.split('\n\n')
sel = [p for p in paras if ('value_bin' in p or 'fine' in p or 'transition' in p or 'mean' in p.lower() and 'bin' in p)]
print('\n---\n'.join(p[:600] for p in sel)[:3500])
print({n: SHAPES[n] for n in NAMES if ('value' in n or 'transition' in n or 'fine' in n)})
print(list(VOCAB.keys()))
def summ(v):
    try: return getattr(v,'shape',None) or (list(v.keys()) if isinstance(v,dict) else type(v).__name__)
    except Exception as e: return str(e)
for nm in ['EDIT','KVn','Won','Bin2','DECL','NF5','RW2','DOUT','QE','TY','a_y','QY']:
    print(nm, summ(globals()[nm]) if nm in globals() else 'MISSING')
with open('/work/notes.md','a') as f:
    f.write("\n## Session 15 (restart 15)\n- Run 277: read value-head part of ARCH, shapes, check build vars. Keep replies short, neutral wording.\n")
```