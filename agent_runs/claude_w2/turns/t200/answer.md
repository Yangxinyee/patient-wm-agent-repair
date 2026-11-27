The atom table has a bundle index column (`bidx`). The collection hook can match each active sequence's bundle to its insulin start atom through it. This run does the stage-wise decodability study described above:
- teacher-forced runs on the 300 stays and 7 dose arms;
- at every bundle that starts an insulin infusion, record u, the prelude output (slot mean), hn (768 numbers), the new state h (768) and out;
- fit cross-validated ridge decoders for the arm's insulin rate and its deviation from the policy dose.

That deviation is computed from the recorded glucose at that hour. For each stage I report factual R² and the arm-change slope.

```python
code_st='''
def collect_states_ins(R_, arms, W, idsel=None):
    A_=R_['A']; codeA=A_.code.values; kindA=A_.kind.values
    insS=np.where((codeA==CODE['insulin'])&(kindA==0))[0]
    sk2j={(int(A_.si.values[j]),int(A_.bidx.values[j])):j for j in insS}
    store=[]
    def hs(k,act_n,h,out,ctx):
        s=ctx['seq_si'][act_n]; m=np.array([(int(x),k) in sk2j for x in s])
        if not m.any(): return
        mi=np.where(m)[0]; n=act_n[mi]
        js=np.array([sk2j[(int(x),k)] for x in s[mi]])
        store.append(dict(n=n,j=js,arm=ctx['seq_arm'][n],si=s[mi],u=ctx['u'][mi].clone(),Ep=ctx['Ep'][mi].mean(1).clone(),
                          hn=ctx['hn'][mi].reshape(len(mi),-1).clone(),y=h[n].reshape(len(mi),-1).clone(),out=out[n].clone()))
    res_=run_rollout_h(R_,W,arms,n_paths=1,seed=0,t_hide=1e9,hook_state=hs)
    C={k_:(np.concatenate([d[k_] for d in store]) if not torch.is_tensor(store[0][k_]) else torch.cat([d[k_] for d in store]).numpy()) for k_ in store[0]}
    return C
'''
open('/work/states_ins.py','w').write(code_st); exec(code_st)
t0_=time.time()
CS=collect_states_ins(RD,armsD,W)
print('rows',len(CS['n']),'time',round(time.time()-t0_))
A_=RD['A']; tt_=A_.t.values
mults=np.stack([arm_mult(RD,a,'insulin') for a in armsD])
rate_arm=A_.rate.values[CS['j']]*mults[CS['arm'],CS['j']]
# policy from recorded glucose at hour floor(t)
gq=A_[(A_.code.isin(GLC))&(np.abs(A_.t-np.round(A_.t))<1e-6)]
gq=gq.assign(h=np.round(gq.t).astype(int), isf=(gq.code==CODE['glucose_finger']).astype(int)).sort_values('isf').groupby(['si','h']).value.last()
hr_=np.floor(tt_[CS['j']]+1e-6).astype(int)
gv=gq.reindex(pd.MultiIndex.from_arrays([CS['si'],hr_])).values
p_=np.where(np.isnan(gv),0.0,PF(np.nan_to_num(gv)))
dev_=rate_arm-p_
grpS=CS['si']; armS=CS['arm']; keyS=pd.Series(list(zip(CS['si'],CS['j'])))
def cv_eval(X,y,alpha=1.0):
    pr=np.zeros(len(y))
    for trn_,tst_ in GroupKFold(5).split(X,y,grpS):
        m=Ridge(alpha=alpha).fit(X[trn_],y[trn_]); pr[tst_]=m.predict(X[tst_])
    r2f=1-np.mean((pr-y)[armS==0]**2)/np.var(y[armS==0])
    df_=pd.DataFrame({'k':keyS,'a':armS,'y':y,'p':pr}); b0=df_[df_.a==0].set_index('k'); sl=[]
    for a in range(1,len(armsD)):
        jn=df_[df_.a==a].set_index('k').join(b0,rsuffix='0',how='inner'); dy=jn.y-jn.y0; dp=jn.p-jn.p0
        sl.append(round(float((dp*dy).sum()/max((dy*dy).sum(),1e-9)),2))
    return round(r2f,3),sl
for nm_ in ['u','Ep','hn','y','out']:
    X=CS[nm_]
    print(nm_, X.shape[1], 'rate',cv_eval(X,rate_arm), 'dev',cv_eval(X,dev_))
print('time',round(time.time()-t0_))
```