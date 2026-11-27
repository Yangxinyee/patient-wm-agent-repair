The insulin sliding scale has the same structure as norepinephrine. Readings of exactly 150, 180, 220, 260 and 300 get a dose halfway between the two tiers (for example 1.55 at 180, between 1.05 and 2.1). So the policy also acts on unrounded values, and my 10-mg/dL cells mix split and pure readings at each tier boundary.

The impulse regressions already controlled for the reading itself, which should absorb most of this. The insulin result (≈0) is therefore probably unaffected. Still, the main readout rests on it.

To be rigorous, I'll rebuild the clean insulin residual the same way:
- exact-integer reading cells;
- the five split readings excluded;
- given doses only.

Then I'll redo the impulse responses and the implied x2 shift. If it stays near zero (within about ±0.5 mg/dL), the null for insulin is final.

```python
Fi=np.where(np.isnan(Fg),-1,np.clip(Fg,100,400)).astype(int)
splitv=np.isin(Fi,[150,180,220,260,300])
gA=(Ins>0)&(Fi>=151)&~splitv
cellI=np.where(gA,Fi-100,0)
cntI=np.bincount(cellI[gA],minlength=302); smI=np.bincount(cellI[gA],weights=Ins[gA],minlength=302); mI=smI/np.maximum(cntI,1)
eAc=np.where(gA,Ins-mI[cellI],0.0); eAc[:,0]=0
print('eAc sd among given %.3f n %d'%(eAc[gA].std(),gA.sum()))
ys=[];x1=[];f1=[];ss=[]
for s in range(1,25):
    si=s+1; ok=gA[:,si]&~np.isnan(Fg[:,si-1])
    ys.append(eAc[ok,si]); x1.append(np.stack([Fg[ok,si-1]/100, Ins[ok,si-1], np.nan_to_num(Lb[ok,si])/100, np.isnan(Lb[ok,si])],1)); f1.append(cellI[ok,si]); ss.append(np.where(ok)[0])
b,V=fe_ols(np.concatenate(ys),np.concatenate(x1),np.concatenate(f1),[],np.concatenate(ss))
print('eAc exog: prevFinger/100 %+.4f(%.4f) prevDose %+.4f(%.4f) lab/100 %+.4f(%.4f)'%(b[0],np.sqrt(V[0,0]),b[1],np.sqrt(V[1,1]),b[2],np.sqrt(V[2,2])))
okIc=lambda si: gA[:,si]
bIc,sIc=impulse(Fg,eAc,okIc,lambda si: cellI[:,si],ctrl_ins,label='INS clean -> finger')
pIc,_=impulse(Ins,eAc,okIc,lambda si: cellI[:,si],ctrl_ins,K=11,label='INS clean -> later dose')
dI_=np.zeros(13)
for k in range(1,13): dI_[k]=bIc[k]-sum(dI_[l]*pIc[k-l] for l in range(1,k))
implied(dI_,sIc,Ins,Fg,label='INSULIN clean direct (deconv)')
bIl,sIl=impulse(Lb,eAc,okIc,lambda si: cellI[:,si],ctrl_ins,label='INS clean -> lab')
implied(bIl,sIl,Ins,Lb,label='INSULIN clean -> lab')
```