On the clean sample (glucose measured at t, insulin always on), all three specifications agree closely. The structural insulin response is about −2.0, −2.45, −2.1, −1.7, −1.3, −1.0, −0.7, −0.5, −0.35, −0.25, −0.25, −0.3 mg/dL per u/h at lags 1–12, with cumulative effect −12.6 to −13.2 by lag 12 and no positive tail. The earlier tail came from mixing in on/off-region and stale-reading rows. I take this sample's response as the main estimate.

Doubling a dose is a big extrapolation from ±35% noise. Next I will check two things at lag 1. First, linearity: whether the effect per u/h is the same in bands centred on 1, 2 and 3 u/h, and whether a quadratic term is needed. Second, whether a log-dose model fits better. Equal slopes across bands would support linear extrapolation to ×2 and ×1/2.

```python
S1 = S.copy(); S1['y']=gf[S1.stay.values, np.minimum(S1.t.values+1,T-1)]; S1=S1[S1.t.values+1<T]
S1['band']=pd.cut(S1.gl0,[155,180,220,260,400],right=False)
for bnd,d in S1.groupby('band'):
    b,sb=wols(d.assign(one=1.0),'y',['xb','gl0','g1','r1'],'one')
    print('band',bnd,'n',len(d),'slope/unit %.2f se %.2f'%(b.xb,sb.xb),' mean r %.2f'%d.r.mean())
S1['lr']=np.log(S1.r.where(S1.r>0)); S1['xl']=S1.lr-S1.groupby('g5').lr.transform('mean')
S1['xb2']=S1.xb**2
b,sb=wols(S1.assign(one=1.0),'y',['xb','xb2','gl0','g02','g1','r1'],'one'); print('quadratic: lin %.2f (%.2f) quad %.2f (%.2f)'%(b.xb,sb.xb,b.xb2,sb.xb2))
for lab,cols in [('linear',['xb']),('log',['xl'])]:
    d=S1.dropna(subset=['xl']).assign(one=1.0); bb,_=wols(d,'y',cols+['gl0','g02','g1','r1'],'one')
    Xm=np.column_stack([np.ones(len(d))]+[d[c].values for c in cols+['gl0','g02','g1','r1']]); e=d.y.values-Xm@bb.values
    print(lab,'coef',round(bb[cols[0]],3),'RSS',round((e**2).sum()/1e6,4))
```