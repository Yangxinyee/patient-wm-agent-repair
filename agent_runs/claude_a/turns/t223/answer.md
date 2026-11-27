The gateless write is dead. Unpenalized, the head-3 decoder reaches magnitudes in the thousands at non-drug bundles. Penalizing the off-target output enough to bound it makes the drug-bundle residual sd rise from 0.29 to 0.54–0.87 units/h, and off-target values still reach 3–34.

Such values in a staging dim would also corrupt the layer norms downstream. So a staging dim needs bounded off-target output in every design, and a gate is unavoidable.

Since the rate targets were wrong when I first tried the no-staging design, that design deserves a retest. I will decode the insulin rate directly from `n3_0` (block-0 FFN input) and from E (prelude output). I will also test two gates:
- insulin present versus not;
- any drug versus none.

For the gates I will count how many non-target samples score above the lowest target score, and how many target samples fall below the highest non-target score.

How I will read it:
- A decoder from `n3_0` with residual sd ≲ 0.3 and slope ≈ 1, plus a separable gate, gives the leanest design: just the two FFN units per drug.
- Otherwise I need a bounded staging with a gate.

```python
def ridge(X, y, lam):
    X1=np.c_[X,np.ones(len(X))]; return np.linalg.solve(X1.T@X1+lam*np.eye(X1.shape[1]), X1.T@y)
def ap(X,w): return np.c_[X,np.ones(len(X))]@w
Eall={k:STS[k]['E'][:,:,keep].numpy() for k in DS}; N3={k:STS[k]['n3_0'][:,:,keep].numpy() for k in DS}
for nm,FE in (('n3_0',N3),('E',Eall)):
    Xi=np.concatenate([FE[k][RS[k]>0].reshape(-1,len(keep)) for k in DS]); yi=np.concatenate([np.repeat(RS[k][RS[k]>0],4) for k in DS])
    wd_=ridge(Xi,yi,1e-1)
    msg=[]
    for k in DS:
        on=RS[k]>0; d_=ap(FE[k][on].reshape(-1,len(keep)),wd_); y_=np.repeat(RS[k][on],4)
        msg.append(f"{k}: sd {np.std(d_-y_):.3f} sl {np.polyfit(y_,d_,1)[0]:.3f}")
    print(nm,'decoder |', ' ; '.join(msg))
    for gname,lab in (('ins',lambda k: RS[k]>0),('drug',lambda k: (RS[k]>0)|isNE)):
        Xg=np.concatenate([FE[k].reshape(-1,len(keep)) for k in DS]); yg=np.concatenate([np.repeat(np.where(lab(k),1.,-1.),4) for k in DS])
        wg_=ridge(Xg,yg,1e-1); s_=ap(Xg,wg_); pos=s_[yg>0]; neg=s_[yg<0]
        print(f"  gate {gname}: pos min {pos.min():.3f} neg max {neg.max():.3f} | #neg>posmin {int((neg>pos.min()).sum())} #pos<negmax {int((pos<neg.max()).sum())} of {len(pos)}/{len(neg)}")
```