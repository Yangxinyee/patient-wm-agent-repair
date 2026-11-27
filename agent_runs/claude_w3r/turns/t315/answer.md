Run 283 confirmed how the value head computes. It is one hidden GELU layer on the code-conditioned input, then a linear map to the 17 value logits. A per-code, per-last-bin transition row is added to those logits. So I can write the head in plain numpy and do not need to inspect kernel code further.

Next I will take stock of what the earlier sessions left in the kernel:
- the core edit dictionary, checked against the tensor shapes,
- the read vectors for the value head,
- the gf-only selector and its margins,
- the value-head shapes, the code ids and the tuning stay list for the forward requests.

The next run will then add the value-head part of the drug-response edit. The plan holds if `EDIT` exists, its shapes match and the read vectors are present. If they are missing, I will rebuild only the value-head stage and use the core pathway as it is.

```python
import numpy as np, json
open('/work/notes.md','a').write("\n## Session 16 (restart 16)\n- Value head: hid=gelu(L(wc,value_bin.0)); logits=L(hid,value_bin.2)+embed_transition.value_bin.weight[code*17+lastbin]. Run 284: inventory of build vars. Keep wording neutral/short.\n")
G = globals()
def A(x):
    if hasattr(x,'detach'): x = x.detach().cpu().numpy()
    return np.asarray(x)
def desc(v):
    try:
        if hasattr(v,'columns'): return f"DF{v.shape}:{list(v.columns)[:20]}"
        if hasattr(v,'shape') and hasattr(v,'dtype'): return f"{type(v).__name__[:5]}{tuple(v.shape)}"
        if isinstance(v, dict): return "{"+",".join(f"{k}:{desc(x)}" for k,x in list(v.items())[:6])+(",.." if len(v)>6 else "")+"}"
        if isinstance(v,(list,tuple)): return f"{type(v).__name__}{len(v)}["+",".join(desc(x) for x in list(v)[:4])+"]"
        if isinstance(v,(int,float,np.floating,np.integer)): return f"{float(v):.4g}"
        return type(v).__name__
    except Exception: return 'ERR'
for n in ['EDIT','QG','QM','QV','WG','CG','RR','DQ','GD','RD','VC','UNITS','GR','GRa','TY']:
    print(n,'=',desc(G[n]) if n in G else 'MISSING')
if 'EDIT' in G:
    for k,v in EDIT.items():
        try:
            o=A(load_original(k)).astype(np.float64); e=A(v).astype(np.float64)
            print(' E', k, e.shape, tuple(SHAPES[k]) if k in SHAPES else 'NOSHAPE', 'dmax %.3g dnorm %.3g'%(np.abs(e-o).max(), np.linalg.norm(e-o)))
        except Exception as ex: print(' E', k, 'ERR', ex)
for k in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.2.weight','embed_transition.value_bin.weight']: print(k, SHAPES[k])
print('VOCAB keys', list(VOCAB.keys()))
c=VOCAB.get('codes'); print('codes', list(c.items())[:14] if isinstance(c,dict) else list(c)[:14])
names=[n for n,v in G.items() if not n.startswith('_') and n not in ('SHAPES','VOCAB','G') and (hasattr(v,'shape') or isinstance(v,dict))]
print(len(names)); print(' '.join(f"{n}{tuple(getattr(G[n],'shape',()))}" for n in names)[:2500])
s1=json.load(open('/work/sel1.json')); print('sel1',type(s1).__name__,len(s1)); print(json.dumps(s1)[:1500])
```