`HT` reads the current saved edits, so the replica heads follow `save_edit` directly.

The readout was normalised on the wrong contrast. I estimated J from edited-versus-original factual states, and that difference also contains the effects of every earlier injection accumulated over the stay. The quantity that matters for a dose edit is the contrast between the x2 run and the factual run of the edited model, and it is only about 0.2× as large, hence the slopes of 0.22 and 0.16.

I'll re-estimate J from that x2-versus-factual contrast on one half of the stays and test the slope on the other half. I'll also measure the readout noise in the edited factual states, refit the context term orthogonal to J, and check the factual-optimal gains. I'm also writing a notes update now.

The fix is confirmed if the held-out slope is about 1 with noise sd well under 1. If the held-out slope is far from 1, the channel response is too context-dependent to read linearly.

```python
with open('/work/notes.md','a') as f: f.write("""
## Session 10 (runs 264-285): CORE DOSE CHANNEL
- Arch: doses enter core ONLY via action token u_k -> prelude.0 attention. Prelude head 2 attends u with weight 1.00 on drug bundles.
- slot residual norm ~800, head2 output ~0.7 -> dose signal swamped (why model ignores dose). E=prelude_norm(slots) cov rank-deficient (102 null dims).
- Edit design: (1) kv rows 288:336 (head2 value) += cp*(pEI x wI2 + pEN x wN2), cp=40, wI2/wN2 = rate probes on ln1(u) (PR2), pEI/pEN = Ve[:,-1/-2];
  (2) core.b_in.weight += tI x rI + tN x rN (rI,rN null-space readers of dE; tI,tN = TC@V2_[:,-1/-2] scaled to SNR 3/6). Vars: WkvF, BinF, KV, BI.
- Replica disturbance at this scale: dnll all +0.0008 (GL +.002, MA +.002). Channel is one-step (decays within ~1h).
- Readout: J from factual edit-vs-orig is wrong (accumulated effects); use cf contrast (x2 vs factual, edited) for J.
- HT(n)=load(n): replica heads read saved edits. heads_full returns (wc (n,4,192), lg (n,4,17)).
""")
FR={}; half=set(stays[:50])
for dr,(cods,dc,rc) in GR.items():
    m_=RW['ev'].isin(cods).values; rr=RW[rc].values[m_]; po=RW['post'].values[m_]&(rr>0); sA=RW['s'].values[m_]
    inA=np.array([x in half for x in sA]); W0=np.concatenate([D3[s]['wc0'] for s in stays])[m_]; W1=WC1[m_]; Wc=WCc[dr][m_]
    dC=(Wc-W1); J_=(dC[po&inA]/rr[po&inA,None]).mean(0)
    Sg=np.cov(W1[~po].T)+CP+1e-6*np.eye(192); g=np.linalg.solve(Sg,J_); g/=g@J_
    slB=np.polyfit(rr[po&~inA],dC[po&~inA]@g,1)[0]; slA=np.polyfit(rr[po&inA],dC[po&inA]@g,1)[0]
    mu_=W1.mean(0); ex=RW[dc].values[m_]; ehat=rr-ex; sel=rr>0
    Pj=np.eye(192)-np.outer(J_,J_)/(J_@J_); Z=(W1-mu_)@Pj; bm=Ridge(alpha=1.0).fit(Z[sel],ehat[sel]); beta=Pj@bm.coef_
    u=(W1-mu_)@(g-beta)-bm.intercept_*sel; res=OB[m_]-PM1[m_]
    print(dr[:3],'slope A/B',round(float(slA),2),round(float(slB),2),'noise sd W1',round(float(((W1[~sel]-mu_)@g).std()),3),'prec',round(float((GP[np.isin(evs40,cods)]@g).std()),3),'corr(u,dev)',round(float(np.corrcoef(u[sel],ex[sel])[0,1]),3))
    for c_ in cods:
        mc=RW['ev'].values[m_]==c_; print('   ',c_[-6:],'A',round(float(np.polyfit(u[mc],res[mc],1)[0]),2),'sd u',round(float(u[mc].std()),3))
    FR[dr]=dict(g=g,beta=beta,b0=float(bm.intercept_),mu=mu_,J=J_)
```