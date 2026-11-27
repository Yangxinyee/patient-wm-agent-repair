The build finished all 1,500 stays in 187 s: 70,738 factual rows over all four codes. 821 stays have insulin running from hour 12 and 569 have NE, each with ×2 and ×½ edit responses.

Before solving for decoders I need the arrays in one place and two feasibility checks:
1. **Gate cost.** The dose response in with_code (Δx = Δq) is identical for glucose and MAP rows at the same time point. So a glucose-only correction needs a nonlinear code gate, and I need to know how separable glucose and MAP rows are along some direction (LDA d′).
2. **Response size.** How large the mean response to each edit is relative to the factual within-(code, last bin) variance. That ratio is the SNR that sets the factual cost.

A large d′ would mean a cheap gate; a modest one would push me to a different gating design.

```python
CODES4=['glucose_finger','glucose_lab','map_arterial','map_cuff']; C4={c_:i_ for i_,c_ in enumerate(CODES4)}
offs=np.cumsum([0]+[len(g_['code']) for g_ in GEN]); N_=offs[-1]
Xf=np.concatenate([g_['X'] for g_ in GEN]).astype(np.float64); cf=np.concatenate([[C4[c_] for c_ in g_['code']] for g_ in GEN])
hf=np.concatenate([g_['hour'] for g_ in GEN]); pf=np.concatenate([g_['prev'] for g_ in GEN]); uf=np.concatenate([g_['cur'] for g_ in GEN])
sf=np.concatenate([[g_['s']]*len(g_['code']) for g_ in GEN]); DQf=np.concatenate([g_['DQ'] for g_ in GEN])
lbf=np.array([0 if np.isnan(p_) else int(lastbin_of(CODES4[c_],p_)) for c_,p_ in zip(cf,pf)])
print('N',N_, 'code counts', np.bincount(cf), 'lb range', lbf.min(), lbf.max())
isg=cf<2; ism=cf>=2
mg_=Xf[isg].mean(0); mm_=Xf[ism].mean(0); Sw=np.cov(np.vstack([Xf[isg]-mg_, Xf[ism]-mm_]).T)
gld=np.linalg.solve(Sw+1e-3*np.eye(192), mg_-mm_); pg=Xf@gld
print('LDA dprime', ((pg[isg].mean()-pg[ism].mean())/np.sqrt(0.5*(pg[isg].var()+pg[ism].var()))).round(2))
print('z-noise sd along LDA', (DQf[:,0,:]@gld).std().round(3), 'within sd', pg[isg].std().round(3))
```