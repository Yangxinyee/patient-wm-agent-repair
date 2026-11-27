The tensor list shows `c_out` is a plain 192×192 linear map and the coda is a full transformer layer (attention + FFN). So every step rewrites `h` nonlinearly, and a hand-built clean channel through `h` would be invasive.

Before choosing between a new channel and exposing an existing signal, I'll check where the random dose deviation lives right after an infusion bundle. I'll regress (dose − expected dose given the glucose reading) on four state parts: `out`, the slot-mean of `h`, the full 4-slot `h`, and z, using grouped CV. Two outcomes:
- High R² in the full `h` but low in `out` / mean: the information exists but the heads can't see it.
- Low R² everywhere: the model discards it and I must write it in myself.

```python
I_,N_,G_=HM
X_=np.asarray(S['x']); sid=np.asarray(S['stay_id']); hrs=np.asarray(S['hour']); hi=np.asarray(S['has_insulin']).astype(bool)
rows_i=[];yv=[];yd=[]
for i in np.where(hi)[0]:
    hh=int(np.floor(hrs[i]+0.01)); key=(int(sid[i]),hh)
    if key in I_ and key in G_ and np.isfinite(G_[key]):
        rows_i.append(i); yd.append(I_[key]); yv.append(I_[key]-cexp(G_[key]))
rows_i=np.array(rows_i); yv=np.array(yv); yd=np.array(yd); print('n insulin bundles', len(rows_i), 'dev sd', yv.std().round(3), 'dose sd', yd.std().round(3))
parts={'out':X_[rows_i,:192],'hmean':X_[rows_i,192:960].reshape(-1,4,192).mean(1),'hfull':X_[rows_i,192:960],'z':X_[rows_i,960:]}
for nm_,Xp in parts.items():
    r2=[]
    for y_ in [yv,yd]:
        pr=cvp(Ridge(alpha=1.0),Xp,y_,groups=sid[rows_i],cv=GroupKFold(5)); r2.append(round(1-np.var(y_-pr)/np.var(y_),3))
    print(nm_, 'R2 dev,dose', r2)
```