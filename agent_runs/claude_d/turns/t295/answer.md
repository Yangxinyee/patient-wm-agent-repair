The E4 head reads the probe injection with the wrong sign. MAP shifts by −0.40 on average, reaching −1.15 at full injection, while the original head gives +0.09. So an arbitrary off-manifold direction is read unpredictably, and building a new reader is too noisy (SNR about 1).

The most reliable carrier is the model's own in-range NE dose axis, which my E4 channel already reads correctly (x1/2 works). The plan is to edit only the encoding of NE START atoms in bin 16, which never occur in the records:
- Their mlp1 output becomes m(rate 1.0) + β·(r′ − 1)·Δm, where Δm is the per-unit-rate in-range dose slope in mlp1-output space.
- So "more than the recorded maximum" continues the in-range dose direction, and β (start at 1) is the calibration knob.

The correction c(r′) = target − original output is a smooth function of z_rate. I fit it with 4 gated mlp1 units (674, 411, 314, 524): a constant, a linear term and two smooth hinges in z, gated by NE START ∧ bin 16. This run builds it and checks offline:
- the fit error of the correction,
- that factual insulin, NE and END encodings barely move,
- that bin-16 encodings match the target,
- leakage margins and bf16 error.

```python
open('/work/notes.md','a').write("""
- Run 256-266: encode_action.mlp1 gates: wGf (NE START & bin16 -> +1, others <=-1), wGp (any NE START), wZ (reads z_rate) exact, orthogonal to numeric/unit/kind embs. Cheap units 674,411,314,524 (ablation max|du|<=.004).
- u(r) for NE is BIN-DOMINATED (step fn), out-of-range moves off-manifold.
- PROBE fwd12 (states both, train selP=/work/selP.json, unit674 injection q*2.5*v1 on all NE START): footprint at MAP_a lag1 93% one direction but reader SNR~1 (natural var); E4 head reads it as MAP -0.4 (wrong sign) -> abandoned new head channel.
- PLAN: bin16 NE START mlp1 output := m(r=1.0) + beta*(r'-1)*dm_dose (in-range slope), 4 gated units; calibrate beta with dose_edit.
""")
def m1(A_,W0=W10,b0=b10,W2=W12,b2=Wt('encode_action.mlp1.2.bias')): return gel(A_@W0.T+b0)@W2.T+b2
rr=np.linspace(.25,1.0,61); Mr=m1(np.array([aNE(r) for r in rr])); dm=np.polyfit(rr,Mr,1)[0]
rp=np.linspace(1.0005,2.15,200); zp_=np.array([zz(r,ed['rate']) for r in rp]); Ap=np.array([aNE(r) for r in rp]); Mo=m1(Ap); m10=m1(aNE(1.0)[None])[0]
def build_enc(beta,units=(674,411,314,524),kap=25.0):
    Mt=m10+beta*(rp-1)[:,None]*dm[None]; Cc=Mt-Mo; zm=1.89
    spec=[(4.0,0.0),(4.0,1.5),(0.0,3.0),(0.0,-3.0)]
    Phi=np.array([gel(c+b*(zp_-zm)) for c,b in spec]).T; V=np.linalg.lstsq(Phi,Cc,rcond=None)[0]
    W0n=W10.copy(); b0n=b10.copy(); W2n=W12.copy(); b2n=Wt('encode_action.mlp1.2.bias').copy()
    for (c,b),j,vj in zip(spec,units,V):
        b2n+=W12[:,j]*act[:,j].mean(); W0n[j]=kap*wGf+b*wZ; b0n[j]=-kap+c-b*zm; W2n[:,j]=vj
    return W0n,b0n,W2n,b2n,Mt,np.abs(Phi@V-Cc).max()
W0n,b0n,W2n,b2n,Mt,fe=build_enc(1.0); print('corr fit err',round(fe,4),'|c| max',np.abs(Mt-Mo).max().round(2))
U_t=np.array([uenc2([m_,aE]) for m_ in []]) if False else None
un=lambda a: uenc2([a,aE],W0=W0n,b0=b0n,W2=W2n,b2=b2n)
du_f=max(np.abs(un(a)-uenc2([a,aE])).max() for a in NEA[:40:3]); du_i=max(np.abs(uenc2([a],W0=W0n,b0=b0n,W2=W2n,b2=b2n)-uenc2([a])).max() for a in INSU[::5]+ENDS)
def u_from_m(mv): s=(mv+m1(aE[None])[0])/np.sqrt(2)+Wt('encode_action.size.weight')[:,0]*np.log(3)+Wt('encode_action.size.bias'); return u_of_s(s)
err_t=max(np.abs(un(aNE(r))-u_from_m(m10+(r-1)*dm)).max() for r in (1.01,1.3,1.7,2.0))
pmax=max(float((np.array(INSU+NEA[:40]+ENDS)@W0n[j]+b0n[j]).max()) for j in (674,411,314,524)); print('factual NE du',round(du_f,4),'ins/END du',round(du_i,4),'bin16 target err',round(err_t,4),'max pre non-target',round(pmax,2))
pb=max(abs(float(bf(W0n[j])@bf(aNE(r))+b0n[j])-float(aNE(r)@W0n[j]+b0n[j])) for j in (674,411,314,524) for r in (1.1,1.5,2.0)); print('bf16 pre err',round(pb,3))
```