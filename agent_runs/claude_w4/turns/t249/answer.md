The v3 value directions are not safe for remapping. They are exactly 0 on all action tokens, including the no-action token, but not on the other tokens head 1 can attend to:

| Token type | projection mean | sd |
|---|---|---|
| slots | ≈ 0.29 | 0.04 |
| event tokens | up to ≈ 0.7 | ≈ 0.3 |
| z tokens | up to ≈ 0.38 | ≈ 0.25 |
| learned initial slots | up to 1.2 | |

Usually that doesn't matter, because those tokens get about 0.04% attention. But at the bundles where head-1 attention to the slots rises to 0.72, a remap could inject junk worth a mg/dL or more. The readout leak problem is separate and worse.

The cached action tokens are rich enough to train proper readouts:
- a 2,773-point grid of rate pairs (UG);
- 3,368 real rate pairs (UR);
- 1,128 combinations (UC);
- the 2,319 recorded drug-bundle tokens.

Next I'll fit the v4 building blocks and measure their quality:
1. **Value directions.** The 48-dim head-1 value directions with the smallest second moment over all token types (action tokens weighted heavily), with their residual projections.
2. **Leak-free readouts.** Ridge readouts on the normalised action token for the centred insulin rate (rate − c when present) and the centred NE rate. They are constrained to ≈0 on slot, event, z, initial-slot and no-action tokens, at several constraint weights. For each I'll report accuracy on recorded and edited doses, plus the attention-weighted leak at drug and non-drug bundles.
3. **Where head-1 attention to the slots is high,** so I know which bundles to protect.

The constraint is achievable if a weight exists with rate rms below about 0.1 and leak below about 0.02 units. If it doesn't, the readout must be restricted further or the attention sharpened.

```python
X_slot=np.concatenate([TOK[:,:4].reshape(-1,192),TOKn[:,:4].reshape(-1,192)]); X_e=np.concatenate([TOK[:,4],TOKn[:,4]]); X_z=np.concatenate([TOK[:,6],TOKn[:,6]])
u_none=np.asarray(TOKn[0,5],dtype=np.float64)
X_init=np.asarray(sl,dtype=np.float64)          # first-bundle slot tokens = 0 + core.slots
def lab_pairs(pairs):
    a=np.array([0. if p[0] is None else float(p[0]) for p in pairs]); b=np.array([0. if p[1] is None else float(p[1]) for p in pairs])
    pa=np.array([p[0] is not None and float(p[0])>0 for p in pairs]); pb=np.array([p[1] is not None and float(p[1])>0 for p in pairs]); return a,b,pa,pb
gI_,gN_,gpI,gpN=lab_pairs(grid_pairs); rI2,rN2,rpI,rpN=lab_pairs(pairs_real)
XU=np.concatenate([UG,UR,Uk,u_none[None]]); yIr=np.concatenate([gI_,rI2,RI,[0]]); yNr=np.concatenate([gN_,rN2,RN,[0]])
pIu=np.concatenate([gpI,rpI,RI>0,[False]]); pNu=np.concatenate([gpN,rpN,RN>0,[False]])
print('u-token fit set', XU.shape, 'ins present', pIu.sum(), 'ne present', pNu.sum(), ' max ins rate', yIr.max(), ' max ne', yNr.max())
cI=float(RI[RI>0].mean()); cN=float(RN[RN>0].mean()); print('centres: mean recorded rate when present: ins %.3f ne %.3f'%(cI,cN))
L_U=LN1(XU); L_O=np.concatenate([LN1(X_slot),LN1(X_e),LN1(X_z),np.repeat(LN1(X_init),200,0)])
print('other-token set', L_O.shape)
# head-1 attention at drug and non-drug bundles (slot queries)
with torch.no_grad(): _,attn_nd=prelude_tail(torch.as_tensor(TOKn,dtype=torch.float32),ret_att=True)
A_d=attc[:,1,:4,:].numpy(); A_n=attn_nd[:,1,:4,:].numpy()
hi_=A_d[:,:,:4].sum(-1).max(1)>0.05; print('drug bundles with head-1 slot-attention>5%%: %d of %d; hours'%(hi_.sum(),len(hi_)), np.round(hrd[hi_][:12],2) if 'hrd' in globals() else '')
print('non-drug bundles slot-att>5%%: %d of %d'%((A_n[:,:,:4].sum(-1).max(1)>0.05).sum(),len(A_n)))
def leak(r, TOKx, A):   # attention-weighted readout from non-u tokens, per bundle (max over slot queries)
    Lx=LN1(np.asarray(TOKx).reshape(-1,192)).reshape(len(TOKx),7,192)@r
    Lx[:,5]=0; return np.abs((A*Lx[:,None,:]).sum(-1)).max(1)
for nm,y_,c_,pm_ in [('ins',yIr,cI,pIu),('ne',yNr,cN,pNu)]:
    yc=np.where(pm_,y_-c_,0.0)
    for wO in [0.,1e-2,1e-1,1.,10.]:
        Aq=L_U.T@L_U+wO*(L_O.T@L_O)+1e-6*np.eye(192); r=np.linalg.solve(Aq,L_U.T@yc)
        e=L_U@r-yc; ld=leak(r,TOK,A_d); ln_=leak(r,TOKn,A_n)
        print(f'{nm} wO={wO:6g}: |r| {np.linalg.norm(r):9.1f} rms err all {np.sqrt((e**2).mean()):.4f} (recorded {np.sqrt((e[len(UG)+len(UR):-1]**2).mean()):.4f}) none-token {float(LN1(u_none)@r):+.4f} | other-token |r.x| rms {np.sqrt(((L_O@r)**2).mean()):.3f} | leak drug median {np.median(ld):.4f} p99 {np.percentile(ld,99):.3f} max {ld.max():.3f}; non-drug max {ln_.max():.3f}')
        globals()[f'r_{nm}_{wO:g}']=r
```