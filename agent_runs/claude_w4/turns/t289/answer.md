The channel works at the E level:
- Along the two chosen directions, the natural normalised prelude output varies with sd ≈ 0.0007.
- An injection of c = 6 shows up in E as exactly c·f/σ (ratio 1.000) with under 1% off-target leakage. The median signal is 0.074 per unit readout, so natural noise amounts to about 0.01 rate units.

One worry for factual fit: centring the insulin channel only on the mean rate may cost likelihood. The band explains about 80% of rate variance, and an uncaptured state-dependent mean adds error. I'll measure this with a factual pass rather than guess.

This run builds v4:
1. Target directions in the new state: the bisector of the finger and lab news directions for insulin (negative sign), and of the arterial and cuff directions for NE (positive).
2. Map each target back through the coda, by inverting it on the observed states and solving with its Jacobian, then through `c_out`, giving the write vector t at the median σ.
3. Re-point head 1's out-projection: insulin readout → c·q_I, NE readout → c·q_N, presence → −c·r̄·q_I.
4. Add rank-1 terms to `b_in` that replace its response along each channel by k·t, with k = 1 for both drugs to start.

I'll then check with the replica that the resulting change in the new state at drug bundles points along the target with slope ≈ k·readout. I'll also check that the predicted change at non-drug bundles is negligible.

```python
gp, bp = L('core.prelude_norm.weight'), L('core.prelude_norm.bias'); spd = np.log1p(np.exp(L('core.delta_in'))); Cout = L('core.c_out.weight'); Wbin = L('core.b_in.weight')
nuF, nuL, nuA, nuC = [nu[j].mean(0) for j in range(4)]
uv = lambda v: v/np.linalg.norm(v)
tgt_I = -0.87*uv(uv(nuF)+uv(nuL)); tgt_N = 4.3*uv(uv(nuA)+uv(nuC))
CP='core.coda.0.'; Wqc, Wkvc, Woc = L(CP+'attn.q.weight'), L(CP+'attn.kv.weight'), L(CP+'attn.out.weight')
def coda(c):
    n1 = lnw(c, CP+'norm1'); Q=n1@Wqc.T; K=n1@Wkvc[:192].T; V=n1@Wkvc[192:].T; outs=[]
    for h in range(4):
        sl=slice(48*h,48*h+48); s=np.einsum('nqd,nkd->nqk',Q[...,sl],K[...,sl])/np.sqrt(48); a=np.exp(s-s.max(-1,keepdims=True)); a/=a.sum(-1,keepdims=True); outs.append(np.einsum('nqk,nkd->nqd',a,V[...,sl]))
    c1 = c + np.concatenate(outs,-1)@Woc.T; return c1 + mlp(lnw(c1, CP+'norm3'), CP+'ffn')
rows = REP['row']; Y = H[rows]; cc = Y.copy()
for it in range(40): cc = Y - (coda(cc) - cc)
print('coda inversion resid', np.abs(coda(cc)-Y).max())
def solve_dc(tgt, eps=1.0):
    dc = tgt.copy()
    for it in range(6):
        dy = (coda(cc + eps*dc[None,None,:]) - coda(cc))/eps; r = tgt - dy.mean((0,1)); dc = dc + r
    dy = (coda(cc + eps*dc[None,None,:]) - coda(cc))/eps
    return dc, np.linalg.norm(tgt - dy.mean((0,1)))/np.linalg.norm(tgt), dy
dcI, eI, dyI = solve_dc(tgt_I); dcN, eN, dyN = solve_dc(tgt_N)
print('coda solve rel err', eI, eN, '|dc|/|tgt|', np.linalg.norm(dcI)/np.linalg.norm(tgt_I), np.linalg.norm(dcN)/np.linalg.norm(tgt_N), 'cos(dc,tgt)', uv(dcI)@uv(tgt_I), uv(dcN)@uv(tgt_N))
print('per-bundle response spread (cos of dy with tgt) I', np.round(np.quantile([uv(v)@uv(tgt_I) for v in dyI.mean(1)],[.05,.5,.95]),3), 'N', np.round(np.quantile([uv(v)@uv(tgt_N) for v in dyN.mean(1)],[.05,.5,.95]),3))
def solve_h(dc, lam=1e-4):
    return np.linalg.solve(Cout.T@Cout + lam*np.eye(192), Cout.T@dc)
dhI, dhN = solve_h(dcI), solve_h(dcN); print('c_out solve rel resid', np.linalg.norm(Cout@dhI-dcI)/np.linalg.norm(dcI), np.linalg.norm(Cout@dhN-dcN)/np.linalg.norm(dcN), '|dh|', np.linalg.norm(dhI), np.linalg.norm(dhN))
sig_ref = np.median(REP['x2'].std(-1)); c = 6.0; print('sigma_ref', sig_ref)
tI = (sig_ref/c)*dhI/spd; tN = (sig_ref/c)*dhN/spd
rbar = tsel[tsel.event=='insulin'].rate.mean(); print('rbar insulin', rbar, ' mean NE rate', tsel[tsel.event=='norepinephrine'].rate.mean())
np.savez('/work/v4_parts.npz', tI=tI, tN=tN, tgt_I=tgt_I, tgt_N=tgt_N, qI=qs[:,0], qN=qs[:,1], rbar=rbar, sig_ref=sig_ref, c=c, A=A, M=M)
def build_v4(kI, kN, cent=True, save=True):
    qI, qN = qs[:,0], qs[:,1]
    TA = c*np.c_[qI, qN, (-rbar*qI if cent else 0*qI)] @ np.linalg.inv(M)
    Wo_new = Wo.copy(); Wo_new[:,48:96] = Wo[:,48:96]@(np.eye(48)-A@A.T) + TA@A.T
    Wb = Wbin.copy()
    for q,t,k in [(qI,tI,kI),(qN,tN,kN)]:
        qE = gp*q; w = q/gp; Wb = Wb + np.outer(k*t - Wbin@qE, w)
    if save:
        save_edit('core.prelude.0.attn.out.weight', torch.tensor(Wo_new, dtype=torch.float32)); save_edit('core.b_in.weight', torch.tensor(Wb, dtype=torch.float32))
        save_edit('core.prelude.0.attn.kv.weight', torch.tensor(kv3, dtype=torch.float32))
    return Wo_new, Wb
import torch
Wo4, Wb4 = build_v4(1.0, 1.0)
# replica check at drug bundles
Xs4d = H[rows-1]*np.exp(-((hr[rows]-hr[rows-1])*3600)[:,None,None]*np.exp(fa)[None,None,:]) + slots[None]
Td = np.concatenate([Xs4d, REP['e'][:,None], REP['u'][:,None], (X[rows,960:992]@zinW.T+zinB)[:,None]],1)
_, x2n = prelude_b(Td, Wkv=kv3, Wo=Wo4); En = lnw(x2n,'core.prelude_norm')
dbin = spd*((En@Wb4.T) - (REP['E']@Wbin.T)); dcr = dbin@Cout.T; dyr = coda(cc+dcr) - coda(cc)
fI = rI - rbar*(rI>0); fN_ = (rN-0.52)*(rN>0)
Bfit = np.linalg.lstsq(np.c_[fI, fN_], dyr.mean(1), rcond=None)[0]
print('replica dy per unit: insulin |b|', np.linalg.norm(Bfit[0]), 'cos with tgt_I', uv(Bfit[0])@uv(tgt_I), '| NE |b|', np.linalg.norm(Bfit[1]), 'cos tgt_N', uv(Bfit[1])@uv(tgt_N))
resid = dyr.mean(1) - np.c_[fI,fN_]@Bfit; print('rel resid', np.linalg.norm(resid)/np.linalg.norm(dyr.mean(1)), ' edited:', edited())
```