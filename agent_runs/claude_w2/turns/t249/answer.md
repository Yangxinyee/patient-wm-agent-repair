The data are in: 123,080 head inputs across 8 arms × 300 stays. Of these, 43k are finger glucose and 13k lab glucose; the rest are MAP readings, which I need for the gate.

Next I'll fit a ridge decoder from the head input to the target correction r on the glucose rows. Test stays are held out. Three numbers decide whether the head-only edit is viable:

- **Held-out R²** of the decoder.
- **Counterfactual slope:** the decoded change between an edited arm and the recorded arm, regressed on the true change. It should be ≈1.
- **Factual error sd:** at gain 1 this becomes noise in the factual predictions. It needs to stay ≲1.3 mg/dL to keep the NLL change within budget.

I'll also look at the spread of the predictive variance. A logit tilt shifts the mean in proportion to that variance, so its spread sets how uneven the effective gain will be.

```python
glu=np.isin(Dw['code'],GLC); r_all=Dw['d'][:,:4]@np.array(a_i)+Dw['p'][:,:4]@np.array(c_i)
stays=np.unique(Dw['si']); rng=np.random.default_rng(0); te=np.isin(Dw['si'],rng.choice(stays,60,replace=False))
def ridge_fit(X,y,alpha):
    xm=X.mean(0); ym=y.mean(); Xc=(X-xm).astype(np.float64)
    w=np.linalg.solve(Xc.T@Xc+alpha*len(y)*np.eye(X.shape[1])/1000,Xc.T@(y-ym)); return w, ym-xm@w
key=pd.DataFrame({'si':Dw['si'],'hj':Dw['hj'],'arm':Dw['arm'],'i':np.arange(len(r_all))})
k0=key[key.arm==0].set_index(['si','hj']).i
def pair_eval(rh,mask):
    out=[]
    for a in range(1,len(armsW)):
        ka=key[(key.arm==a)&mask[key.i.values]].set_index(['si','hj']).i; j=ka.index.intersection(k0.index)
        ia=ka.loc[j].values; i0=k0.loc[j].values; dr=r_all[ia]-r_all[i0]; dh=rh[ia]-rh[i0]; m=np.abs(dr)>0.5
        if m.sum()>20: out.append(round(float(np.polyfit(dr[m],dh[m],1)[0]),2))
    return out
for al in [0.01,0.1,1.0]:
    w,b0=ridge_fit(Dw['wc'][glu&~te],r_all[glu&~te],al); rh=Dw['wc']@w+b0
    m0=glu&te&(Dw['arm']==0); mt=glu&te
    print('alpha',al,'R2 test',round(r2(r_all[mt],rh[mt]),3),'fact R2',round(r2(r_all[m0],rh[m0]),3),'fact err sd',round(float(np.std(rh[m0]-r_all[m0])),2),'r sd',round(float(np.std(r_all[m0])),2),'cf slopes',pair_eval(rh,glu&te))
Pg=Dw['P'][glu&(Dw['arm']==0)]; cg=CENT[Dw['code'][glu&(Dw['arm']==0)]]; p_=Pg[:,1:]/Pg[:,1:].sum(1,keepdims=True)
mu=(p_*cg).sum(1); sd=np.sqrt((p_*cg**2).sum(1)-mu**2); print('pred sd quantiles', np.quantile(sd,[.05,.25,.5,.75,.95]).round(1))
```