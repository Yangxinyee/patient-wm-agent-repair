Restoring `G_ = F.gelu` from /work/wm2.py worked exactly: `tfnll(W0z)` and the gain-1 insulin ×2 readout both match their earlier values. The gain sweep then gave a clear negative result:

| Gain | Insulin ×2 | Insulin ×½ | Norepinephrine ×2 | Norepinephrine ×½ | TF NLL |
|---|---|---|---|---|---|
| 3 | −3.66 | +2.08 | +0.68 | −0.99 | 1.52984 |
| 6 | −3.60 | +2.27 | +0.70 | −0.93 | 1.52964 |
| 10 | −3.62 | +2.23 | +0.71 | −0.90 | 1.52920 |

The answers barely move as the gain rises from 3 to 10. The small changes are lopsided: ×½ grows a little while ×2 does not move. This is what the attention problem from run 281 predicts. The injected direction also shifts the attention split between the u token and its copy, so a large positive or negative dose residual just moves attention onto one token and the effect saturates.

So the existing directions in `VD_` cannot be scaled up. I need new directions with two properties:
1. They change nothing about the attention scores at drug bundles.
2. They change the prelude output E the way a real dose change does in the original model.

I plan to get these from a regularised least-squares fit over many drug bundles, using finite-difference Jacobians of E with respect to the copy token. That needs the exact prelude code and the data formats of the existing helpers (`E_of`, `collect_drug`, the saved bundle context `c`).

So this run is a cheap look at those formats: the `bstep` prelude code in wm2.py, and the structures `GATE`, `DEC2`, `PH`, `Pk` and `ue_` that `mkW3` reads. If the formats are clear, the next run computes the new directions and tests them directly.

```python
s_=src_all['wm2.py']; L_=s_.split('\n')
i0=[i for i,l in enumerate(L_) if l.startswith('def bstep(')][0]
i1=[i for i in range(i0+1,len(L_)) if L_[i].startswith('def ')][0]
print('\n'.join(l[:160] for l in L_[i0:i1]))
for l in L_:
    if l.startswith(('def mha','def bln','def lnorm','def blin','def bself')): print(l[:200])
for f_n in ['E_of','collect_drug','base_of','copyK','copyz']:
    co=globals()[f_n].__code__; print(f_n, 'names', co.co_names[:30], 'vars', co.co_varnames[:20])
print('c', {k_:tuple(getattr(v_,'shape',())) for k_,v_ in c.items()} if isinstance(c,dict) else type(c))
print('Pk',Pk.shape,'ue_',type(ue_).__name__, getattr(ue_,'shape',None), 'K',K)
for d_ in ['ins','ne']:
    print(d_, 'GATE', [(type(x).__name__, getattr(x,'shape',x)) for x in GATE[d_]], 'DEC2', DEC2[d_].shape, 'PH', PH[d_].shape, 'VD_', VD_[d_].shape, np.linalg.norm(VD_[d_]))
```