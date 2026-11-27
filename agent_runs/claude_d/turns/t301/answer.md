The candidate bundle holds exactly one insulin START atom (rate 1.0, the stay's first insulin record) and nothing else. So the mismatch must come from fields the insulin atom carries that NE START atoms do not. There are candidates:
- `kq` has 4 rows, all with nonzero gate dot products (up to 0.64).
- `action`, `route` and `status` each have one nonzero row (dots 0.31, −0.10, −0.12).

If insulin atoms carry kq[2] plus action[1], my gate value on insulin would rise by about 1. At the chosen κ ≈ 4.4 that erodes the −4.5 margin to about 0, so the correction would leak onto insulin.

This run searches unit × rate_unit × kq × action × route × status for the combination that reproduces the z posterior on this bundle. It then validates the top candidates on all 15 insulin-only bundles. A match near 1e-4 settles the insulin atom formula.

```python
CAT={n:Wt(f'embed.categorical.{n}.weight') for n in ['action','route','status']}; KQ=Wt('embed.kq.weight')
import itertools
def aIN2(r,u_,ru,kq,ac,rt,stt):
    v=aIN(r,u_,ru)
    if kq is not None: v=v+KQ[kq]
    for nm,ix in (('action',ac),('route',rt),('status',stt)):
        if ix is not None: v=v+CAT[nm][ix]
    return v
def zerr_b(rw,f):
    g=TRi[TRi.stay_id==rw.stay_id]; S_=g[np.round(g.hour*60).astype(int)==rw.minute]; r=float(S_.rate.iloc[0])
    c=ctx_of(int(rw.prev_row),float((rw.minute-rw.prev_min)*60)); zt=XALL[int(rw.row),960:992]
    return float(np.abs(np.asarray(zpo(np.asarray(c).reshape(-1),e_empty,uenc2([f(r)]))).reshape(-1)[:32]-zt).max())
combos=list(itertools.product(range(6),range(4),[None,0,1,2,3],[None,0,1],[None,0,1],[None,0,1]))
r0=cand.iloc[0]; sc=[(zerr_b(r0,lambda r,c=c: aIN2(r,*c)),c) for c in combos]; sc.sort(key=lambda t:t[0]); print('bundle0 top', [(round(a,5),c) for a,c in sc[:5]])
cl=[c for _,c in sc[:5]]; full=[(round(float(np.mean([zerr_b(rw,lambda r,c=c: aIN2(r,*c)) for _,rw in cand.iterrows()])),5),c) for c in cl]; print('all15', full)
```