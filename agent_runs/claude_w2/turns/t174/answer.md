Collection took 19 s and gave 208k head rows: 300 stays × 14 arms, recorded readings, doses changed per arm. Next I test the core hypothesis on glucose rows. I compute
- the model's own teacher-forced response M = pm(arm) − pm(factual) per dose lag, which gives its interventional one-step coefficients;
- the required correction T = S − M, where S is the structural one-step response from my data OLS (−1.99, −1.47, −0.89, +0.17, …);
- a ridge decoder of T from the head input wc, fitted on 240 stays and tested on 60.

A high held-out R², a test slope near 1 and a small spurious output on factual rows would confirm the decoder route. A low R² would refute it.

```python
nA=len(armsC); tau=np.round(DC['t']).astype(int)
def lagmat(Glist,D,L=12):
    X=np.zeros((len(D['n']),L))
    for a in range(len(Glist)):
        m=np.where(D['arm']==a)[0]
        for k in range(1,L+1):
            ix=tau[m]-k+3; ok=ix>=0
            X[m[ok],k-1]=Glist[a][D['si'][m[ok]],ix[ok]]
    return X
Gins=[dose_grid(RC,'insulin',arm_mult(RC,a,'insulin')) for a in armsC]
Gne=[dose_grid(RC,'norepinephrine',arm_mult(RC,a,'norepinephrine')) for a in armsC]
DinsA=lagmat(Gins,DC); DneA=lagmat(Gne,DC)
key=DC['si'].astype(np.int64)*10**6+DC['j']
f0=pd.Series(np.where(DC['arm']==0)[0],index=key[DC['arm']==0])
fidx=f0.reindex(key).values.astype(int)
dIns=DinsA-DinsA[fidx]; dNe=DneA-DneA[fidx]; Mresp=DC['pm']-DC['pm'][fidx]
GL=np.isin(DC['code'],[CODE['glucose_finger'],CODE['glucose_lab']]); MP=np.isin(DC['code'],[CODE['map_arterial'],CODE['map_cuff']])
ins_arms=[1,2,3,4,5,6,12]; ne_arms=[7,8,9,10,11,13]
mI=GL&np.isin(DC['arm'],ins_arms)
bm=np.linalg.lstsq(dIns[mI],Mresp[mI],rcond=None)[0]
print('model interventional one-step coef (glucose on ins lags):',np.round(bm,2))
beta_ols=np.array([-1.99,-1.47,-0.89,.17,.81,.65,.40,.32,.23,.03,0,0])
print('data one-step OLS coef                                   :',beta_ols)
# model response to NE on glucose
mN=GL&np.isin(DC['arm'],ne_arms); print('glucose resp to NE lags:',np.round(np.linalg.lstsq(dNe[mN],Mresp[mN],rcond=None)[0],2))
mNm=MP&np.isin(DC['arm'],ne_arms); bmn=np.linalg.lstsq(dNe[mNm],Mresp[mNm],rcond=None)[0]; print('MAP resp to NE lags (model):',np.round(bmn,2))
mIm=MP&np.isin(DC['arm'],ins_arms); print('MAP resp to ins lags (model):',np.round(np.linalg.lstsq(dIns[mIm],Mresp[mIm],rcond=None)[0],3))
# wc
def wc_of(D,W,idx,chunk=40000):
    outs=[]
    for s0 in range(0,len(idx),chunk):
        b=idx[s0:s0+chunk]; bt=torch.as_tensor(b)
        _,wc=head_value_logits(D['base'][bt],torch.as_tensor(D['code'][b],dtype=torch.long),D['f8'][bt],D['lb'][bt],torch.zeros(len(b),dtype=torch.long),W,D['zmu'][bt],1)
        outs.append(wc)
    return torch.cat(outs).numpy().astype(np.float64)
iG=np.where(GL&~np.isin(DC['arm'],ne_arms))[0]
WCg=wc_of(DC,W,iG)
Tg=(dIns[iG]@beta_ols)-Mresp[iG]
stays_=np.unique(DC['si']); test_s=set(stays_[::5])
te=np.array([s in test_s for s in DC['si'][iG]]); trn=~te
def ridge(X,y,alpha):
    mu=X.mean(0); Xc=X-mu; A_=Xc.T@Xc+alpha*np.eye(X.shape[1]); w=np.linalg.solve(A_,Xc.T@(y-y.mean())); return w,y.mean()-mu@w
for alpha in [1.0,10.0,100.0]:
    w_,b_=ridge(WCg[trn],Tg[trn],alpha); pr=WCg@w_+b_
    for nm_,mm in [('train',trn),('test',te)]:
        fa=mm&(DC['arm'][iG]==0); iv=mm&(DC['arm'][iG]!=0)
        sl=np.polyfit(Tg[iv],pr[iv],1)[0]; r2=1-np.var(Tg[iv]-pr[iv])/np.var(Tg[iv])
        print(f'alpha {alpha:6.1f} {nm_}: interv R2 {r2:.3f} slope {sl:.3f} | factual pred sd {pr[fa].std():.2f} mean {pr[fa].mean():.2f} | T sd {Tg[iv].std():.2f}')
```