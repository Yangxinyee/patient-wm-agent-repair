The existing head-2 value readout is weak. The code-8 rate reaches a cross-validated R² of 0.55, and code 9 only 0.26. Doubling the code-8 rates raises the decoded value only 1.43-fold, a slope of 0.19 against 2r. So head 2's current value projection does not carry the rate linearly.

Next I'll check whether the full 192-dimensional infusion encoding u itself carries the rate linearly and responds proportionally to doubling. If it does, one repurposed value dimension in head 2 can read it. If u itself saturates, I'll have to add a linear path through the encoder of the infusion atoms.

```python
U=[];U2=[];RR=[];RR2=[]
for s in sids:
    s=int(s); B=stay_bundles(s)
    for k in range(1,len(B)):
        g=B[k]
        if not ((g.code==8)|(g.code==9)).any(): continue
        e,u,_,_=enc_bundle(g); U.append(np.asarray(u,dtype=np.float64)); RR.append(rts(g))
        if rts(g)[0]>0:
            e2,u2,_,_=enc_bundle(dbl(g)); U2.append(np.asarray(u2,dtype=np.float64)); RR2.append(rts(g)[0])
U=np.array(U);U2=np.array(U2);RR=np.array(RR);RR2=np.array(RR2)
for c in [0,1]:
    p=cvp(Ridge(alpha=1e-2),U,RR[:,c],cv=5); print('u code',8+c,'cv R2 %.3f'%(1-((p-RR[:,c])**2).mean()/RR[:,c].var()))
m=Ridge(alpha=1e-2).fit(U,RR[:,0]); d1=m.predict(U[RR[:,0]>0]); d2=m.predict(U2)
print('x2 ratio %.2f slope %.2f'%(d2.mean()/d1.mean(), np.polyfit(2*RR2,d2,1)[0]))
print('n_atoms per action bundle', np.unique([len(B) for B in [stay_bundles(int(sids[0]))[k] for k in range(1,6)]]))
```