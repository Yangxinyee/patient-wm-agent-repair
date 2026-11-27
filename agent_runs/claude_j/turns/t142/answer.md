The maps are nested (`map[stay][hour]`). Insulin bundles sit at hour h+0.083, separate from the glucose reading at hour h, so each infusion bundle holds only drug atoms. I'll redo the regression of the dose deviation (dose − expected dose given the glucose reading at the same hour) and of the raw dose on each state part. The question is the same: is the random part of the dose present in the state right after the infusion bundle?

```python
rows_i=[];yv=[];yd=[]
for i in np.where(hi)[0]:
    s_=int(sid[i]); hh=int(np.floor(hrs[i]+0.01))
    if hh in I_.get(s_,{}) and hh in G_.get(s_,{}):
        rows_i.append(i); yd.append(I_[s_][hh]); yv.append(I_[s_][hh]-cexp(G_[s_][hh]))
rows_i=np.array(rows_i,dtype=int); yv=np.array(yv,float); yd=np.array(yd,float)
print('n', len(rows_i), 'dev sd', yv.std().round(3), 'dose sd', yd.std().round(3))
parts={'out':X_[rows_i,:192],'hmean':X_[rows_i,192:960].reshape(-1,4,192).mean(1),'hfull':X_[rows_i,192:960],'z':X_[rows_i,960:]}
for nm_,Xp in parts.items():
    r2=[]
    for y_ in [yv,yd]:
        pr=cvp(Ridge(alpha=1.0),Xp,y_,groups=sid[rows_i],cv=GroupKFold(5)); r2.append(round(1-np.var(y_-pr)/np.var(y_),3))
    print(nm_, 'R2 dev,dose', r2)
```