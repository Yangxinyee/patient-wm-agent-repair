The linear route from the prelude to the next state is weak. The transfer matrix's top singular value is 0.3, against 0.03 for a random direction, and its median is 0.006. Most of what the model writes into the state passes through the nonlinear blocks, not the direct b_in path.

What matters, though, is not the size of the transmitted signal but its precision at the head, i.e. signal relative to the head input's natural variation along the readout. A small signal along a direction that varies little in factual data can still be read precisely.

The cleanest test asks directly whether adding the channel makes the lag-1 insulin dose more decodable at the head than in the unedited model. The baseline is factual R² 0.63, with arm slopes 0.57–0.85. I'll try three write directions: the top two singular directions of the transfer and a random one. The amplitude is c = 15 per U/h, about 10% of the residual norm at 3 U/h. For each I'll report:
- the cross-validated decodability of the lag-1 dose from the edited model's with_code (factual R² and per-arm slopes, small ridge);
- the factual value NLL side effects by code.

If one direction lifts factual R² well above 0.63 (say 0.9 or more) with slopes about 1 and little side effect, the channel works. If decodability barely moves, the core blurs the channel and I'll go back to a calibrated head-only edit.

```python
cands_v={'top1':Vt_[0],'top2':Vt_[1],'rand':vs}
for k_ in cands_v:
    v_=cands_v[k_]-cands_v[k_].mean(); cands_v[k_]=v_/np.linalg.norm(v_)
gB=np.where(np.isin(DB['code'],GLC))[0]
wcB,_,_=wc_hidden(DB,W,gB); XB=wcB.numpy()
aB=DB['arm'][gB]; sB=DB['si'][gB]; kB=pd.Series(list(zip(DB['si'][gB],DB['j'][gB])))
dB_,pB_,devB_=lag_feats(RB,DB,arms3,PF,gB)
def dec_eval(X,y,alpha=1e-3):
    pr=np.zeros(len(y))
    for trn_,tst_ in GroupKFold(5).split(X,y,sB):
        m_=Ridge(alpha=alpha).fit(X[trn_],y[trn_]); pr[tst_]=m_.predict(X[tst_])
    r2f=1-np.mean((pr-y)[aB==0]**2)/np.var(y[aB==0]); errf=np.std((pr-y)[aB==0])
    df_=pd.DataFrame({'k':kB,'a':aB,'y':y,'p':pr}); b0=df_[df_.a==0].set_index('k'); sl=[]
    for a in [1,2]:
        jn=df_[df_.a==a].set_index('k').join(b0,rsuffix='0',how='inner'); dy=jn.y-jn.y0; dp=jn.p-jn.p0
        sl.append(round(float((dp*dy).sum()/max((dy*dy).sum(),1e-9)),3))
    return round(r2f,3),round(errf,3),sl
for al in [1e-3,1e-1]:
    print('baseline alpha',al,'d1',dec_eval(XB,dB_[:,0],al),'dev1',dec_eval(XB,devB_[:,0],al))
for k_,v_ in cands_v.items():
    Wc_=make_W_chan(W,wI2,v_,15.0)
    RC_,_,DCk=cf_collect(tr, idsD, arms3, Wc_, seed=0)
    nC=nll_of(DCk['P'][fB],obB[fB])
    per={CNAME[c]:round(float(nC[DB['code'][fB]==c].mean()-nB[DB['code'][fB]==c].mean()),5) for c in [CODE[m] for m in MEAS]}
    wcC_,_,_=wc_hidden(DCk,W,gB); XC=wcC_.numpy()
    print(k_,'dNLL all %.5f'%(nC.mean()-nB.mean()),per)
    for al in [1e-3,1e-1]: print('   alpha',al,'d1',dec_eval(XC,dB_[:,0],al),'dev1',dec_eval(XC,devB_[:,0],al))
```