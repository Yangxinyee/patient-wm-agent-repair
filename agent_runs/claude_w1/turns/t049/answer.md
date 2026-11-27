Two useful facts:
- **Insulin:** it is only ever recorded in hours that also have a glucose reading. The dose is set from that reading, so the hour-t insulin record always follows the hour-t reading.
- **Norepinephrine:** it is sometimes changed without a fresh MAP reading.

The state rows split into reading bundles (fraction 0) and drug bundles (0.083). The 100 bundles at 0.167 carry no drug and are probably one administrative event per stay.

Next I'll test the crucial premise of the readout plan. Is the insulin "innovation" linearly decodable from the model's state just before each finger glucose reading? Innovation means recorded rate minus the rate the dosing rule implies from that hour's reading, estimated non-parametrically on the 30,000 training stays.
- If cross-validated R² is high for the last one to three hours, the plan is viable.
- If R² is near zero, the state doesn't carry the information, and I'd have to edit upstream instead.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
def add_pol(P):
    P=P.copy(); P['Gany']=P.glucose_finger.fillna(P.glucose_lab); P['D']=P.insulin.fillna(0)
    P['prev_on']=(P.groupby(level=0)['D'].shift(1).fillna(0)>0).astype(int)
    P['gb']=(P.Gany//2)*2; return P
Ptr2=add_pol(Ptr); Ptu2=add_pol(Ptu)
tab=Ptr2[Ptr2.Gany.notna()].groupby(['gb','prev_on'])['D'].mean()
key=pd.MultiIndex.from_arrays([Ptu2.gb,Ptu2.prev_on])
Ptu2['pi']=np.where(Ptu2.Gany.notna(), tab.reindex(key).values, 0.0)
Ptu2['pi']=Ptu2['pi'].fillna(0); Ptu2['nu']=Ptu2.D-Ptu2.pi
print('policy R2 on tuning:', 1-np.var(Ptu2.nu[Ptu2.Gany.notna()])/np.var(Ptu2.D[Ptu2.Gany.notna()]))
# state before each finger reading
X_=z['x']; sid_=z['stay_id']; hr_=z['hour']
rowsI=[]
for s_ in np.unique(sid_):
    ii=np.where(sid_==s_)[0]; hs=hr_[ii]
    gf=Ptu2.loc[s_,'glucose_finger']
    for t in gf.index[gf.notna()]:
        if t<2: continue
        j=ii[hs<t]
        if len(j)==0: continue
        rowsI.append((s_,t,j[-1]))
RI=pd.DataFrame(rowsI,columns=['sid','t','row'])
for k in (1,2,3,4):
    RI[f'D{k}']=[Ptu2.D.get((s_,t-k),0) for s_,t in zip(RI.sid,RI.t)]
    RI[f'nu{k}']=[Ptu2.nu.get((s_,t-k),0) for s_,t in zip(RI.sid,RI.t)]
hI=np.array([-2.08,-2.41,-2.16,-1.79])
RI['F']=sum(hI[k-1]*RI[f'nu{k}'] for k in (1,2,3,4))
Xs=X_[RI.row.values]; Xs=(Xs-Xs.mean(0))/(Xs.std(0)+1e-6)
gkf=GroupKFold(5)
def cvr2(Xm,y):
    pr=np.zeros_like(y)
    for a,b in gkf.split(Xm,y,RI.sid):
        m=RidgeCV(alphas=np.logspace(-1,4,12)).fit(Xm[a],y[a]); pr[b]=m.predict(Xm[b])
    return 1-np.mean((y-pr)**2)/np.var(y)
print('n',len(RI))
for c in ['D1','nu1','D2','nu2','nu3','F']:
    y=RI[c].values.astype(float)
    print(c,'sd',round(y.std(),3),'R2 out',round(cvr2(Xs[:,:192],y),3),'R2 all',round(cvr2(Xs,y),3))
```