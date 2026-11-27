Run 240 confirms that the stored counterfactual rows did carry shifted readings. The recent glucose readings sat 12.85 mg/dL lower and the doses were doubled, yet the model's prediction fell only 1.64. That would make sense only if one of two things holds. Either the model barely follows its own recent readings, or doubled doses push its predictions up strongly, through a learned "high dose means high glucose" association.

The two need different fixes, so next I will measure persistence directly. I will shift every glucose reading from hour 13 on down by 13 mg/dL, in both the bundles and the head's last-value inputs, keep the doses as recorded, and see how far the replica's glucose_finger predictions fall.

A fall of about 8–13 mg/dL means persistence is strong and the doubled dose is cancelling it (confounding). A fall of only about 1–2 means the model barely carries recent readings forward. Then the rollout effect of any one-step edit would also fail to compound.

```python
print('age sample',RTs[s0][['minute','has_last','age','last_val']].iloc[5:8].values.tolist())
def shifted(s,dv=-13.0,m0=780):
    B=[(bm,[(a[0],a[1],a[2]._replace(value=a[2].value+dv)) if (a[0].startswith('glucose') and bm>=m0) else a for a in at]) for bm,at in BUN[s]]
    rt=RTs[s].copy(); lm=rt.minute-rt.age
    msk=rt.event.str.startswith('glucose')&(lm>=m0)&(rt.has_last>0); rt.loc[msk,'last_val']=rt.loc[msk,'last_val']+dv
    return B,rt,msk.sum()
st30=stays100[:30]; dd=[]
for s in st30:
    B0,R0=BUN[s],RTs[s]; Bm,Rm,nm=shifted(s)
    BUN[s],RTs[s]=Bm,Rm; CACHE.clear(); a1=amp_eval(1.0,[s])[0]; BUN[s],RTs[s]=B0,R0; CACHE.clear(); a0=amp_eval(1.0,[s])[0]
    ev=R0.event.values; mi=R0.minute.values; c=CEN['glucose_finger']
    m1=torch.softmax(a1[3],-1).mean(1).numpy()[:,1:]@c; m0_=torch.softmax(a0[3],-1).mean(1).numpy()[:,1:]@c
    sel=(ev=='glucose_finger')&(mi>=840)&(mi<1500); dd+=list((m1-m0_)[sel])
print('reading shift -13 -> GF pred change mean %.2f sd %.2f n%d'%(np.mean(dd),np.std(dd),len(dd)))
```