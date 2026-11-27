The averaging over slots is not the problem: every slot, and their mean, decode the rate with the same R² (0.61) and slopes of about 1. All the loss appears at the output LayerNorm.

That is suspicious, because a LayerNorm only rescales per dimension and divides by the row's spread. That spread varies only about 4% here and is uncorrelated with the rate. A more likely cause is my ridge penalty (alpha = 1). The pre-norm state has coordinates around ±48, while the normalised vectors (`out`, with_code) are about 1, so the same penalty shrinks the post-norm decoders far more. That would attenuate their slopes artificially.

I'll redo the `out` and head-level (with_code) decoding with alpha = 1e-4 to 10.
- If the slopes recover to about 1 at small alpha, the head already sees a nearly linear dose signal. A head-only edit is then viable, and the remaining issue is noise, not attenuation.
- If the slopes stay low, the LayerNorm really is distorting the signal.

```python
for al in [1e-4,1e-2,1.0,10.0]:
    print('alpha',al,'out rate',cv_eval(CS['out'],rate_arm,alpha=al))
gD=DD['si'][gidx]; aD=DD['arm'][gidx]; kD=pd.Series(list(zip(DD['si'][gidx],DD['j'][gidx])))
def cv_eval2(X,y,alpha=1.0):
    pr=np.zeros(len(y))
    for trn_,tst_ in GroupKFold(5).split(X,y,gD):
        m=Ridge(alpha=alpha).fit(X[trn_],y[trn_]); pr[tst_]=m.predict(X[tst_])
    r2f=1-np.mean((pr-y)[aD==0]**2)/np.var(y[aD==0])
    df_=pd.DataFrame({'k':kD,'a':aD,'y':y,'p':pr}); b0=df_[df_.a==0].set_index('k'); sl=[]
    for a in range(1,len(armsD)):
        jn=df_[df_.a==a].set_index('k').join(b0,rsuffix='0',how='inner'); dy=jn.y-jn.y0; dp=jn.p-jn.p0
        sl.append(round(float((dp*dy).sum()/max((dy*dy).sum(),1e-9)),2))
    return round(r2f,3),sl,pr
Xw=wcD.numpy()
for al in [1e-4,1e-2,1.0]:
    r1=cv_eval2(Xw,dD[:,0],al); r2=cv_eval2(Xw,dD[:,1],al); r3=cv_eval2(Xw,sK1,al); r4=cv_eval2(Xw,devD[:,0],al)
    print('alpha',al,'wc d1',r1[:2],'d2',r2[:2],'sK1',r3[:2],'dev1',r4[:2])
```