The value head is a plain MLP: with_code (192) → 192 GELU units → 17 logits, plus a code×last-bin table (embed_transition.value_bin). Bin 0 is the no-value slot and bins 1–16 have code-specific centres. The code embedding in with_code is shared with the encoder, so I will not touch it. The edit will live in attr_heads.value_bin (and embed_transition if a constant needs cancelling). That keeps every non-value output unchanged in teacher-forced scoring.

Next I test whether the head input can linearly decode the dose deviation, since that is all the edit can read. For a 300-stay insulin subset, I'll collect head inputs at every glucose reading. This is teacher-forced, under the recorded doses plus six counterfactual dose arms (x2, x1/2 and x0 from hour 12, x3 from 6, x1.5 from 16, and random per-infusion multipliers). Then I fit ridge decoders, cross-validated by stay, from two inputs:
- with_code (linear);
- the 192 hidden activations.

The targets are the dose and policy at lags 1–3, the deviations, and the kap = 1 combined signal. Besides R2, the critical number is the slope of the decoded arm-minus-recorded change against the true change: that is what the rollout will see.
- If dev lag 1 and the combined signal decode with R2 well above 0.5 and a counterfactual slope near 1, a head-only edit is feasible.
- If the slope is far below 1, the state does not carry dose changes linearly. Then I would need to strengthen the dose pathway upstream.

```python
code_dec='''
def lag_abs(DG,a,si,tau,L=12):
    X=np.zeros((len(si),L))
    for k in range(1,L+1):
        ix=tau-k+3; ok=ix>=0
        X[ok,k-1]=DG[a[ok],si[ok],ix[ok]]
    return X
def wc_hidden(D_, W, idx):
    base=D_['base'][idx]; codes=torch.as_tensor(D_['code'][idx],dtype=torch.long)
    lg,wc=head_value_logits(base,codes,D_['f8'][idx],D_['lb'][idx],torch.zeros(len(idx),dtype=torch.long),W,D_['zmu'][idx],1)
    h1=F.gelu(F.linear(wc,W['attr_heads.value_bin.0.weight'],W['attr_heads.value_bin.0.bias']))
    return wc,h1,lg
def lag_feats(R_, D_, arms, pfn, idx, L=12, drug='insulin'):
    DG=dose_stack(R_, arms, drug)
    hk=make_dev_oracle(R_, DG, np.zeros(L), t_hide=1e9, pf=pfn)
    si=D_['si'][idx]; a=D_['arm'][idx]; tau=np.round(D_['t'][idx]).astype(int)
    dev=hk.devs(D_['n'][idx], si, a, tau, np.zeros((1,1),int), L)
    d=lag_abs(DG, a, si, tau, L)
    return d, d-dev, dev
'''
open('/work/dec.py','w').write(code_dec); exec(code_dec)
idsD=list(idsI2)
RD=prep_rollout(tr, idsD)
rngm=np.random.default_rng(5)
armsD=[{}, {'ins':2.0,'ins_from':12.0}, {'ins':0.5,'ins_from':12.0}, {'ins':0.0,'ins_from':12.0}, {'ins':3.0,'ins_from':6.0}, {'ins':1.5,'ins_from':16.0},
       {'ins_mult': np.exp(rngm.normal(0,0.6,len(RD['A'])))}]
t0_=time.time()
RD,resD,DD=cf_collect(tr, idsD, armsD, W, seed=0)
gidx=np.where(np.isin(DD['code'],GLC))[0]
wcD,h1D,lgD=wc_hidden(DD,W,gidx)
dD,pD,devD=lag_feats(RD,DD,armsD,PF,gidx)
print('rows',len(gidx),'time',round(time.time()-t0_))
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
grp=DD['si'][gidx]; armv=DD['arm'][gidx]
G1=G_from_K(Kins,1.0); sK1=devD@G1
targets={'d1':dD[:,0],'p1':pD[:,0],'dev1':devD[:,0],'dev2':devD[:,1],'dev3':devD[:,2],'sK1':sK1}
def cvpred(X,y,alpha=1.0):
    pr=np.zeros(len(y))
    for trn_,tst_ in GroupKFold(5).split(X,y,grp):
        m=Ridge(alpha=alpha).fit(X[trn_],y[trn_]); pr[tst_]=m.predict(X[tst_])
    return pr
key=pd.Series(list(zip(DD['si'][gidx],DD['j'][gidx])))
for fn_,X in [('wc',wcD.numpy()),('h1',h1D.numpy()),('wc+h1',np.hstack([wcD.numpy(),h1D.numpy()]))]:
    for tn,y in targets.items():
        pr=cvpred(X,y)
        r2f=1-np.mean((pr-y)[armv==0]**2)/np.var(y[armv==0]); r2a=1-np.mean((pr-y)**2)/np.var(y)
        df_=pd.DataFrame({'k':key,'a':armv,'y':y,'p':pr}); b0=df_[df_.a==0].set_index('k')
        sl=[]
        for a in range(1,len(armsD)):
            ba=df_[df_.a==a].set_index('k'); jn=ba.join(b0,rsuffix='0',how='inner'); dy=jn.y-jn.y0; dp=jn.p-jn.p0
            sl.append(round(float((dp*dy).sum()/max((dy*dy).sum(),1e-9)),2))
        print(fn_,tn,'R2 fact %.2f all %.2f'%(r2f,r2a),'cf slope by arm',sl)
print('time',round(time.time()-t0_))
```