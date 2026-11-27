Head 2 alone gives an insulin channel with an error of about 0.5 U/h at drug bundles, pooled across all arms including the large x3 rates, with an arm slope of about 1. At bundles without drugs its output stays near zero (mean 0.05, 99th percentile 0.6). Adding head 0 does not help, because its output at those bundles conflicts with the zero target.

A 0.5 U/h error is affordable. Through a lag-1 effect of about 1.6 mg/dL per U/h it adds roughly 0.8 mg/dL of tilt noise, about 0.002 nats per glucose reading. So I'll build the channel on head 2 as a rank-1 addition to the prelude attention output, c·v·(w·o). Nothing is repurposed, so the model's existing computation is untouched apart from the added component.

This run puts the edit into a copy of the weights. v is a random mean-free unit direction, and c is set so a typical factual rate (3 U/h) adds about 5% of the slot-residual norm (≈ 442). I'll run teacher-forced on the 300 stays with recorded doses, x2 and x1/2, and measure:
1. **Transmission:** how the head's input at glucose rows changes relative to the unedited model. I regress that change on the lag-1, lag-2 and lag-3 doses to get the transmitted direction per lag, its consistency (R²) and how fast it decays.
2. **Side effects:** the factual value NLL change for all measured codes when nothing reads the channel yet. It should be about 0.

If the change is mostly explained by lag-1 dose along a stable direction, a head readout can be built on it. If the NLL changes noticeably on its own, I'll need a less disruptive direction.

```python
wI2=np.linalg.solve(np.vstack([XdI[:,48:],o02n.reshape(-1,96)[:,48:]]).T@np.vstack([XdI[:,48:],o02n.reshape(-1,96)[:,48:]])+1e-4*np.eye(48),
                    np.vstack([XdI[:,48:],o02n.reshape(-1,96)[:,48:]]).T@np.r_[ydI,np.zeros(o02n.reshape(-1,96).shape[0])])
np.save('/work/wI_head2.npy',wI2)
rs=np.random.default_rng(11); vs=rs.standard_normal(192); vs-=vs.mean(); vs/=np.linalg.norm(vs)
def make_W_chan(W_,w48,vdir,c,head=2):
    W2_=dict(W_); full=np.zeros(192); full[48*head:48*head+48]=w48
    W2_['core.prelude.0.attn.out.weight']=W_['core.prelude.0.attn.out.weight']+torch.as_tensor(c*np.outer(vdir,full),dtype=torch.float32)
    return W2_
cch=7.0
Wc=make_W_chan(W,wI2,vs,cch)
arms3=armsD[:3]
t0_=time.time()
RB,resB,DB=cf_collect(tr, idsD, arms3, W, seed=0)
RC_,resC_,DC=cf_collect(tr, idsD, arms3, Wc, seed=0)
print('time',round(time.time()-t0_))
# observed bins for all meas rows
def obs_bins(D_):
    ob=np.zeros(len(D_['code']),int)
    for c in [CODE[m] for m in MEAS]:
        m=D_['code']==c
        if not m.any(): continue
        edges=np.array(NUMD[(CNAME[c],'value')]['fine_bin_edges']); v=D_['value'][m]
        ob[m]=np.where(np.isnan(v),0,1+np.searchsorted(edges,np.nan_to_num(v),side='left'))
    return ob
obB=obs_bins(DB)
fB=(DB['arm']==0)&~np.isnan(DB['value'])
nB=nll_of(DB['P'][fB],obB[fB]); nC=nll_of(DC['P'][fB],obB[fB])
print('factual value NLL all meas rows: base %.5f chan %.5f diff %.5f (n=%d)'%(nB.mean(),nC.mean(),nC.mean()-nB.mean(),fB.sum()))
for c in [CODE[m] for m in MEAS]:
    m=DB['code'][fB]==c; print('  ',CNAME[c],m.sum(),'dNLL %.5f'%(nC[m].mean()-nB[m].mean()), 'dpm mean %.3f sd %.3f'%((DC['pm'][fB][m]-DB['pm'][fB][m]).mean(),(DC['pm'][fB][m]-DB['pm'][fB][m]).std()))
# transmission to head wc at glucose rows
gB=np.where(np.isin(DB['code'],GLC))[0]
wcB,_,_=wc_hidden(DB,W,gB); wcC,_,_=wc_hidden(DC,W,gB)
dW=(wcC-wcB).numpy()
dB_,pB_,devB_=lag_feats(RB,DB,arms3,PF,gB)
L3=dB_[:,:3]
coefT,_,_,_=np.linalg.lstsq(np.c_[L3,np.ones(len(L3))],dW,rcond=None)
predT=np.c_[L3,np.ones(len(L3))]@coefT
print('transmission: |D_lag1| %.3f |D_lag2| %.3f |D_lag3| %.3f |const| %.3f; R2 of dwc by lags1-3 %.3f; |dwc| mean %.3f'%(
    np.linalg.norm(coefT[0]),np.linalg.norm(coefT[1]),np.linalg.norm(coefT[2]),np.linalg.norm(coefT[3]),1-((dW-predT)**2).sum()/((dW-dW.mean(0))**2).sum(),np.linalg.norm(dW,axis=1).mean()))
print('cos(D1,D2) %.3f cos(D1,D3) %.3f'%(coefT[0]@coefT[1]/np.linalg.norm(coefT[0])/np.linalg.norm(coefT[1]),coefT[0]@coefT[2]/np.linalg.norm(coefT[0])/np.linalg.norm(coefT[2])))
```