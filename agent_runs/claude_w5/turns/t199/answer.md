The choice of direction barely matters. The best signal-to-variation ratio is 2.3 and the median 1.6, so the natural variation of n3 is close to isotropic.

That makes me worry about a magnitude limit:
- Every read of the state goes through a LayerNorm.
- The only gated write route available, prelude → `prelude_norm` → `b_in`, passes through E, which has a fixed norm.
- If the belief shifts needed in h_new are large compared with what `b_in` can produce from a bounded change in E, the prelude route cannot deliver them.

Before designing further, at a sample of drug bundles I will measure the per-slot sizes of: hf, E, sp·b_in(E), the blocks' increment, h_new and y. I will also measure the largest gain of sp·b_in, and the belief shift for a 10 mg/dL reading change projected onto `b_in`'s range.

If the needed shift of about 8 per slot needs |ΔE| much smaller than |E| ≈ 14, the prelude route is viable.

```python
sp_=Fn.softplus(W64['core.delta_in']); Wb=W64['core.b_in.weight']
SB=torch.linalg.svdvals(sp_[:,None]*Wb); print('sp*b_in sv max %.2f med %.2f min %.3f'%(float(SB[0]),float(SB.median()),float(SB[-1])))
mags=[]
for d in [d for d in D if isdrug(d) and 'vg' in d][::25]:
    r=RK[d['sid']][d['k']]; e,u=enc2(bundles_B[d['sid']][d['k']][1])
    o,yy,zz,E,hf,hn=cstep_h(r['out0'],r['h0'],r['dt'],e,u,r['Mv'],r['Ma'])
    bE=sp_*(E@Wb.T); inc=hn-0.0-(torch.exp(-sp_*torch.exp(W64['core.a_in']))*hf)-bE
    mags.append([float(x.norm(dim=-1).mean()) for x in [hf,E,bE,inc,hn,yy]])
print('per-slot norms hf,E,sp*bin(E),blocks inc,hn,y:',np.round(np.mean(mags,0),1))
v10=10*mvg; print('|10*mvg| per slot %.2f'%float(v10.norm(dim=-1).mean()))
U,S_,Vh=torch.linalg.svd(sp_[:,None]*Wb); x_=torch.linalg.lstsq(sp_[:,None]*Wb,v10[0][:,None]).solution
print('|dE| needed via b_in for 10mg/dL (lstsq) %.2f, resid %.3f'%(float(x_.norm()),float(((sp_[:,None]*Wb)@x_-v10[0][:,None]).norm())))
```