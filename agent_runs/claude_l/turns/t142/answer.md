The tie is confirmed at both edges. The NE rate 0.35 equals an edge, and the dose 1.68 sits against an edge stored as 1.6799999999999997. My current bins give rate 7 and dose 8, which is wrong.

I'll try the other three combinations on the 15 bad bundles:
- rate 8, dose 8 (float32 edges compared with float64 values);
- rate 7, dose 7 (pure float32);
- rate 8, dose 7.

The one that gives about 1e-6 error defines the rounding convention for all my bins.

```python
bad=np.where(errU2>1e-4)[0]
def try_combo(rbv, dbv):
    errs=[]
    for i in bad:
        r=recs[i]; atoms=[fix_atom(a) for a in BUN[r[0]][r[1]][1]]
        ua=[]
        for a in atoms:
            if a['act'] and a['kind']!=2:
                a=dict(a)
                if a['code']==9 and a['rb']==7 and abs(a['num'][4]+0.19)<0.02: a['rb']=rbv; a['db']=dbv
                ua.append(a)
        u=set_enc(W0,ua,'encode_action'); errs.append(np.abs(Pu@u.detach().numpy()-r[5]).max())
    return np.round(np.max(errs),6)
for rbv,dbv in [(7,8),(8,8),(7,7),(8,7)]: print('rb',rbv,'db',dbv, try_combo(rbv,dbv))
```