The inventory shows more than my notes say. lib4.py has `build_ins` and `build_ins2`, so I had started building the insulin pathway. The kernel also holds the role scalars jI=67, jR=2, jS=4, jU=4, plus several full parameter dicts (P1, P2, PK, PP, PS, TJ, …). To work out what runs 187–246 built without printing long source, I'll compare each stored parameter dict with the original weights and list only the names of tensors that differ. I'll also print just the function signatures and comment lines from lib3 and lib4. If P1 or P2 turns out to be a near-complete insulin edit (changes to the prelude out-projection, b_in, a_in, flow.a and the value MLP), I'll evaluate it next. Otherwise I'll rebuild with build_ins2.

```python
import re
for f in ['lib3.py','lib4.py']:
    L=open('/work/'+f).read().splitlines()
    keep=[l.strip()[:110] for l in L if l.strip().startswith('def ') or l.strip().startswith('#')]
    print(f, ' | '.join(keep)[:900])
def diffnames(Pd):
    out=[]
    for k in PO:
        a,b=Pd[k],PO[k]
        if a.shape!=b.shape or not torch.equal(a,b): out.append(k)
    return out
for nm in ['P','P_','P1','P2','PK','PP','PS','TJ','V','W','Wt','D_','Po']:
    try:
        d=diffnames(globals()[nm]); s=[x.replace('core.','c.').replace('weight','w').replace('bias','b') for x in d]
        print(nm, len(d), ','.join(s)[:300])
    except Exception as ex: print(nm,'err',str(ex)[:80])
print({k:globals()[k] for k in ['jI','jR','jS','jU'] if k in globals()}, 'ROLE' in globals() and ROLE)
```