The replica checks out. Attention matches my notebook exactly: head 0 puts 0.946 on the slots, and heads 1, 2 and 3 put 0.998, 0.997 and 0.992 on u. The v3 key/value edit is rank 3:
- one component is mainly insulin rate (fit error 0.11 units/h);
- one is mainly centred NE rate (error 0.025);
- one is insulin presence (error 0.005);
- the original model's head-1 values are about zero along all three (sd 0.001), including at the no-action token.

So I can keep v3's readouts and only re-point the out-projection of these three directions. The mixing matrix M lets me undo the rotation.

Next I choose the channel direction q (one for insulin, one for NE). It should be a slot-token direction in which the normalised prelude output E hardly varies naturally. I'll compute E for all bundles: exactly at drug bundles, and at measurement bundles approximately, dropping the reading token (weight ≈ 0.1). I'll take the low-variance eigen-directions, constrained to have zero mean and to be orthogonal to the mean token and to β/γ, then pick the ones the prelude FFN is least sensitive to.

At drug bundles I'll then inject c = 6 along q and measure two things. The signal is w·ΔE per unit readout, expected near 6/σ ≈ 0.15. The noise is the natural spread of w·E. If the noise sd is below about 0.015 and the off-target share of ΔE is small, the channel is clean.

```python
def prelude_b(T, Wkv=Wkv, Wo=Wo):
    N1 = lnw(T, P+'norm1'); Q = N1[:,:4]@Wq.T; K = N1@Wkv[:192].T; V = N1@Wkv[192:].T; outs=[]
    for h in range(4):
        sl=slice(48*h,48*h+48); s = np.einsum('nqd,nkd->nqk', Q[...,sl], K[...,sl])/np.sqrt(48)
        a = np.exp(s-s.max(-1,keepdims=True)); a/=a.sum(-1,keepdims=True); outs.append(np.einsum('nqk,nkd->nqd', a, V[...,sl]))
    x1 = T[:,:4] + np.concatenate(outs,-1)@Wo.T; return x1, x1 + mlp(lnw(x1, P+'norm3'), P+'ffn')
isdrug = np.zeros(len(sid),bool); isdrug[REP['row']] = True
mrows = np.where(prev_ok & ~isdrug)[0]
Xs_m = H[mrows-1]*np.exp(-((hr[mrows]-hr[mrows-1])*3600)[:,None,None]*np.exp(fa)[None,None,:]) + slots[None]
Tm = np.concatenate([Xs_m, np.repeat(u_const[None,None],len(mrows),0), (X[mrows,960:992]@zinW.T+zinB)[:,None]],1)
x1m, x2m = prelude_b(Tm)
gp, bp = L('core.prelude_norm.weight'), L('core.prelude_norm.bias')
Nall = np.concatenate([ln(x2m), ln(REP['x2'])],0).reshape(-1,192); nbar = Nall.mean(0)
Cn = np.cov(Nall.T); Cd = np.cov(ln(REP['x2']).reshape(-1,192).T)
Cons = np.stack([np.ones(192)/np.sqrt(192), nbar/np.linalg.norm(nbar), (bp/gp)/np.linalg.norm(bp/gp)])
evn, Un = np.linalg.eigh(Cn + 1e3*Cons.T@Cons); print('eig n\'\' smallest', np.format_float_scientific(evn[0],2), np.round(evn[[10,30,60,100,140]],7), '#<1e-5', (evn<1e-5).sum(), '#<1e-4', (evn<1e-4).sum())
Z = Un[:, :30]
W1g = L(P+'ffn.0.weight')*L(P+'norm3.weight')[None,:]; S = W1g.T@W1g
es, Us = np.linalg.eigh(Z.T@S@Z); qs = Z@Us[:, :2]; print('FFN sens of chosen', np.round(es[:4],3), 'median over Z', np.round(np.median(np.diag(Z.T@S@Z)),3))
for j in range(2):
    q = qs[:,j]; print(f'q{j}: var all {q@Cn@q:.2e} drug {q@Cd@q:.2e} | cons dots', np.round(Cons@q,6))
# signal test at drug bundles: add c*q*readout to x1 (all 4 slots), recompute FFN + LN
c = 6.0; x1d = REP['x1']; Ed = REP['E']
for j in range(2):
    q = qs[:,j]; w = q/gp; f = np.c_[rI, (rN-0.52)*(rN>0)][:, j]
    x1p = x1d + c*f[:,None,None]*q[None,None,:]; x2p = x1p + mlp(lnw(x1p, P+'norm3'), P+'ffn'); Ep = lnw(x2p, 'core.prelude_norm'); dE = Ep - Ed
    sigd = REP['x2'].std(-1); sig_pred = (w@dE.transpose(0,2,1)) ; m = np.abs(f)>0.2
    ratio = (sig_pred[m]/(c*f[m,None]/sigd[m])).ravel()
    qE = gp*q; off = dE - (dE@w)[...,None]*qE[None,None,:]
    nat_d = (Ed@w); nat_m = (lnw(x2m,'core.prelude_norm')@w)
    print(f'chan{j}: signal/(c f/sigma) mean {ratio.mean():.3f} sd {ratio.std():.3f}; |offtarget|/|dE| median {np.median(np.linalg.norm(off[m],axis=-1)/np.linalg.norm(dE[m],axis=-1)):.3f}; natural w.E drug mean {nat_d.mean():.4f} sd {nat_d.std():.4f}, meas mean {nat_m.mean():.4f} sd {nat_m.std():.4f}; c/sigma median {np.median(c/sigd):.3f}')
np.savez('/work/chan.npz', qs=qs, Z=Z, evn=evn, nbar=nbar)
```