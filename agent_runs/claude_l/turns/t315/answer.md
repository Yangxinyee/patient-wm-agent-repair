Run 287 gave me the exact core-step code. The slot outputs E come from one self-attention layer (`core.prelude.0`) over 7 tokens — [hf + slots ×4, e, u, z-token] — followed by `prelude_norm`. `E_of(W, h, dt, e, u, zt)` recomputes E for any z-token, so I can get the sensitivity of E to the z-token by finite differences.

The global `c` holds one drug bundle's prelude inputs (hf, e, u, ctx, info). `CAP` (777 items) and `CAPT` (798) are probably lists of such bundles. Each `GATE` entry is a 192-vector plus two scalars; `DEC2` and `PH` are linear maps with a bias (193 numbers each).

Before computing new directions I need to confirm the formats:
- what a `CAP` item and its `info` contain, including how to tell insulin from norepinephrine bundles;
- what `keygrads(c)` returns;
- whether `E_of` accepts batches when given hf with dt = 0;
- the `bself` code;
- how faithfully the copy model (`mkW3` at gain 0) reproduces u in the z-token.

One outcome would let the next run build the new directions directly: a near-exact copy (cosine above 0.99) and a batched `E_of`. A poor copy would itself explain part of the problem.

```python
L_=src_all['wm2.py'].split('\n'); i0=[i for i,l in enumerate(L_) if l.startswith('def bself(')][0]
print('\n'.join(l[:170] for l in L_[i0:i0+8] if l.strip()))
print('CAP', len(CAP), type(CAP[0]).__name__, 'CAPT', len(CAPT), type(CAPT[0]).__name__)
c0=CAP[0]
show=lambda d: {k_:(type(v_).__name__, tuple(getattr(v_,'shape',()))) for k_,v_ in d.items()} if isinstance(d,dict) else repr(d)[:300]
print('CAP0', show(c0)); print('CAPT0', show(CAPT[0]))
print('info c:', repr(c['info'])[:200], '| CAP0 info:', repr(c0.get('info') if isinstance(c0,dict) else None)[:200], '| CAP5 info:', repr(CAP[5].get('info') if isinstance(CAP[5],dict) else None)[:200])
try:
    kg=keygrads(c); print('keygrads ->', type(kg).__name__, getattr(kg,'shape',None) if not isinstance(kg,(list,tuple)) else (len(kg), [getattr(x,'shape',None) for x in kg[:3]]))
except Exception as ex: print('keygrads fail', repr(ex)[:200])
Wc0=mkW3(0.0,0.0,W0z)
def ztok(W,ctx,e,u): return blin(W,bmlp(W,torch.cat([ctx,e,u],-1),'z_post')[:,:32],'z_in')
tt=lambda x: (x if torch.is_tensor(x) else torch.tensor(np.asarray(x))).float()
try:
    hf_=tt(c['hf'])[None]; e_=tt(c['e'])[None]; u_=tt(c['u'])[None]; ctx_=tt(c['ctx'])[None]
    zc=ztok(Wc0,ctx_,e_,u_); print('copy check: cos', float(F.cosine_similarity(zc,u_)), 'rel err', float((zc-u_).norm()/u_.norm()))
    E1=E_of(Wc0,hf_,0.0,e_,u_,zc); E1b=E_of(Wc0,hf_.repeat(3,1,1),0.0,e_.repeat(3,1),u_.repeat(3,1),zc.repeat(3,1))
    print('E_of shape', tuple(E1.shape), 'batched ok', tuple(E1b.shape), float((E1b[0]-E1[0]).abs().max()))
    for d_ in ['ins','ne']:
        gd,t1,t2=GATE[d_]; gdt=torch.tensor(gd).float()
        print(d_,'gate thr',t1,t2,'u.g e.g ctx.g:', [ (round(float(tt(cc['u'])@gdt),2), round(float(tt(cc['e'])@gdt),2), round(float(tt(cc['ctx'])@gdt),2)) for cc in CAP[:4]])
except Exception as ex: print('fail', repr(ex)[:300])
```