This settles the route. A unit change in the z token moves the next finger-glucose prediction by only about 0.1 mg/dL (0.025 for MAP), though the direction is consistent across states. The effect also fades to about 11% by lag 2 and about 5% by lag 3. Effects of tens of mg/dL for a doubled dose therefore cannot come through the model's own value head: the token would have to move far beyond its normal size of about 14.

So the design has to be:
- **Core side:** inject a small residual signal through the z token. "Small" means linear and effectively invisible to the native heads.
- **Head side:** decode that signal from the view and amplify it there. The z_prior/z_head adapter and one value-head unit would tilt the glucose or MAP bins.
- **Memory:** for effects beyond lag 1, `z_post` would read the signal back from context and re-inject it at each step.

All of this depends on one quantity: how decodable an injected signal is in the view, relative to the view's natural variation. I'll compute:
- the mean lag-1 and lag-2 view Jacobians and their spread across states;
- the native view covariance from teacher-forced views on the 100 tuning stays;
- the generalised signal-to-noise eigenvalues of Jᵀ Σ⁻¹ J.

If the best direction gives an SNR per unit token norm of well over 10, with a gain stable across states, the design is workable. If the SNR is low or the gain varies a lot from state to state, the decoded signal would be noisy and the loop would amplify that noise.

```python
VW=[]
for s in BUN:
    R,_=tf_stay(W_cur,BUN[s],keep_view=True)
    seen=set()
    for r in R:
        if r['k'] not in seen: seen.add(r['k']); VW.append(r['view'])
VW=np.array(VW); mu_v=VW.mean(0); Sv=np.cov(VW.T)
ev_,evec_=np.linalg.eigh(Sv); print('views',VW.shape,'view cov eig min/median/max %.2e %.2e %.2e'%(ev_[0],np.median(ev_),ev_[-1]))
def Jlag(L):
    return [r['J_view'] for J in JAC for r in J['res'] if round(r['lag'])==L]
for L in [1,2]:
    Js=np.array(Jlag(L)); Jm=Js.mean(0)
    rel=np.mean([np.linalg.norm(Jx-Jm)/np.linalg.norm(Jm) for Jx in Js])
    sv=np.linalg.svd(Jm,compute_uv=False)
    Sreg=Sv+1e-6*np.eye(192); Sinv=np.linalg.inv(Sreg)
    Mq=Jm.T@Sinv@Jm; w_,V_=np.linalg.eigh(Mq)
    print('lag',L,'n',len(Js),'rel dev of J across states %.2f'%rel,'top sv',np.round(sv[:5],3))
    print('   SNR per unit token norm (sqrt top eig):',np.round(np.sqrt(np.maximum(w_[::-1][:6],0)),1))
    if L==1:
        dbest=V_[:,-1]; sig=np.array([Jx@dbest for Jx in Js]); dec=Sinv@Jm@dbest
        g=np.array([dec@sx for sx in sig])/(dec@(Jm@dbest)); noise=np.sqrt(dec@Sv@dec)/(dec@(Jm@dbest))
        print('   best dir: per-state decoded gain mean %.3f sd %.3f ; native noise in signal units %.4f'%(g.mean(),g.std(),noise))
        J1m=Jm; DEC1=dec; DB1=dbest
```