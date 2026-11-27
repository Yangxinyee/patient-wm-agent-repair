Run 267's numbers are contaminated, and the error is mine. `load()` and my helper defaults were picking up the probe edit still saved in `encode_action.mlp1`. So the "factual NE Δu 2.7" was measured against the probe, not the original, and unit 674's compensation was counted twice. The correction itself fits only roughly: max error 0.17 against corrections up to 17.6.

The real concern is bf16. With κ = 25 the gated pre-activations err by up to 0.43 in bf16. That would make the out-of-range encoding noticeably biased in rollouts. The gate needs a much smaller κ·|w|, which means less margin spent on the z-dependence: insulin z-scores reach 10 under insulin x2.

This run:
- drops the probe edits and recomputes everything against explicit original weights,
- checks how well 2 units (constant + linear in z) or 3 units fit the correction,
- searches gate constraint designs (more negative values for insulin, in-range NE and END atoms) for the smallest bf16 error that still meets the margins.

```python
for n in ['encode_action.mlp1.0.weight','encode_action.mlp1.0.bias','encode_action.mlp1.2.weight','encode_action.mlp1.2.bias']: drop_edit(n)
b12o=load_original('encode_action.mlp1.2.bias').numpy().astype(np.float64); print('edited now', edited())
def m1(A_,W0=W10,b0=b10,W2=W12,b2=b12o): return gel(A_@W0.T+b0)@W2.T+b2
Mr=m1(np.array([aNE(r) for r in rr])); dm=np.polyfit(rr,Mr,1)[0]; Mo=m1(Ap); m10=m1(aNE(1.0)[None])[0]; zm=1.89
Cc=m10+(rp-1)[:,None]*dm[None]-Mo; print('|c| max',np.abs(Cc).max().round(2),'|dm|',np.linalg.norm(dm).round(2))
for nm,spec in [('2u',[(4,0),(4,1)]),('3u',[(4,0),(4,1),(0,2)]),('3u-b',[(4,0),(4,1),(0,-2)])]:
    Phi=np.array([gel(c+b*(zp_-zm)) for c,b in spec]).T; V=np.linalg.lstsq(Phi,Cc,rcond=None)[0]; print(nm,'fit err max',np.abs(Phi@V-Cc).max().round(3),'|V|',np.linalg.norm(V,axis=1).round(1))
def gate(bl,c8,k2):
    t=[c8,0.0]+[0,0,k2]+[0]*nE+[bl]*16+[0.5]+[bl]*16+[0.5]+[0]+[0,0,0,0,0,0]+[0]+[0]*(len(UN)+len(RU))
    Mc2=np.array(CONS[:len(t)]); return np.linalg.lstsq(Mc2,np.array(t),rcond=None)[0]
for bl,c8,k2 in [(-.5,-2,-2),(-1.5,-8,-4),(-1.0,-4,-3),(-1.5,-4,-4)]:
    w=gate(bl,c8,k2); kap=max(8.03/(1-2*bl), 17.1/(-c8)); W_=kap*w+1.0*wZ
    e=max(abs(float(bf(W_)@bf(aNE(r)))-float(W_@aNE(r))) for r in (1.05,1.4,1.8,2.1))
    print(f'bl {bl} c8 {c8} k2 {k2}: |w| {np.linalg.norm(w):.2f} kap {kap:.2f} kap|w| {kap*np.linalg.norm(w):.1f} bf16 err {e:.3f} G(NE16) {float(w@aNE(1.5)):.2f} G(NE.9) {float(w@aNE(.9)):.2f} G(ins16) {float(w@aIN(8,0,0)):.2f}')
```