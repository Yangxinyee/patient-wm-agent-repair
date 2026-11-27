Both decoders and both gates work:
- **Insulin gate:** separates cleanly (positives ≥ 0.785, negatives ≤ −0.571).
- **NE gate:** separates with a smaller margin (positives ≥ −0.002, negatives ≤ −0.60). I'll threshold at the midpoint and use a large gain.
- **Insulin decoder** (f·rate − pibar from [ctx, u]): CV R² 0.91, improving as the ridge penalty shrinks, so I'll try smaller ones. At recorded doses its error sd is 0.49 u/h. My estimate is that this costs little factual NLL at an injection gain near 2.
- **NE rate decoder** (from u): CV R² 0.96, error sd 0.15.

Now I'll build the whole edit in one run:
1. **Directions:** least-squares token directions from the saved Jacobians. v_I lowers next glucose by 1 per unit with MAP unchanged; v_N raises next MAP by 1 per unit with glucose unchanged. I'll check them for nonlinearity with `next_pred` at a realistic scale.
2. **Decoders:** refit both with the best small ridge penalty.
3. **Gated units:** repurpose 4 dead z-posterior units as two gated pairs, GELU(M·gate + decoder) − GELU(M·gate). This is exactly the decoder at drug bundles and 0 elsewhere, written into z dims 18 (insulin) and 28 (NE). The dims' old outputs are removed and their mean is folded into the z_in bias.
4. **Initial gains:** set z_in columns 18 and 28 to 2.0·v_I and 2.5·v_N, near the factual-optimal lag-1 effects.
5. **Check and save:** verify locally that all other z dims are unchanged and the new dims equal the decoders, then save.

```python
Jt=d['Jtok']
def dirfit(rows,tgt,wts):
    A=np.array([wts[c]*Jt[k,:,c] for k in rows for c in range(4)]); b=np.array([wts[c]*tgt[c] for k in rows for c in range(4)])
    G=A.T@A; return np.linalg.solve(G+1e-3*np.trace(G)/192*np.eye(192),A.T@b)
rowsI=np.where(hi_[kAll])[0]; rowsN=np.where(hn_[kAll])[0]
vI=dirfit(rowsI,[-1,-1,0,0],[1,.6,1,.6]); vN=dirfit(rowsN,[0,0,1,1],[1,.6,1,.6])
for nm,v,rows,sc in [('I',vI,rowsI,4.0),('N',vN,rowsN,2.0)]:
    resp=np.einsum('kdc,d->kc',Jt[rows],v); sh=[]
    for k in kAll[rows[:6]]:
        b0=np.asarray(next_pred(int(k)),dtype=float)
        try: b1_=np.asarray(next_pred(int(k),ztok_add=torch.as_tensor(sc*v,dtype=torch.float32)),dtype=float)
        except Exception: b1_=np.asarray(next_pred(int(k),ztok_add=(sc*v).astype(np.float32)),dtype=float)
        sh.append(b1_-b0)
    print(nm,'|v|',np.linalg.norm(v).round(3),'lin resp',resp.mean(0).round(2),'sd',resp.std(0).round(2),'| next_pred shift x',sc,np.mean(sh,0).round(2),np.std(sh,0).round(2))
print('I dec:'); lI=cvfit(XI,yI,S3a[cI],[0.003,0.01,0.03],fI_==1); wI,bI0=ridge(XI,yI,lI)
print('N dec:'); lN=cvfit(UN,yN,S3a[cN],[0.01,0.03],fN_==1); wN,bN0=ridge(UN,yN,lN)
CtxG=np.r_[ctxA[cI],ctxA[cN],ctxA[neg]]
def Mreq(s,t,dec):
    mid=(s[t>0].min()+s[t<0].max())/2; sp=s-mid; p=t>0; n=~p
    req=max(((5-np.minimum(dec[p],0))/sp[p]).max(), ((5+np.maximum(dec[n],0))/(-sp[n])).max()); return mid,1.5*req
sIg=Ug@GATE['I'][0]+GATE['I'][1]; sNg=Ug@GATE['N'][0]+GATE['N'][1]
midI,MI=Mreq(sIg,tIg,np.c_[CtxG,Ug]@wI+bI0); midN,MN=Mreq(sNg,tNg,Ug@wN+bN0); print('midI,MI',round(midI,3),round(MI,1),'midN,MN',round(midN,3),round(MN,1))
W1o=np.asarray(load_original('z_post.0.weight'),dtype=np.float64); b1o=np.asarray(load_original('z_post.0.bias'),dtype=np.float64)
W2o=np.asarray(load_original('z_post.2.weight'),dtype=np.float64); b2o=np.asarray(load_original('z_post.2.bias'),dtype=np.float64)
Wzo=np.asarray(load_original('z_in.weight'),dtype=np.float64); bzo=np.asarray(load_original('z_in.bias'),dtype=np.float64)
U4=dead[:4]; aI,bIu,aN,bNu=[int(j) for j in U4]; am=act[okA][:,U4].mean(0)
W1=W1o.copy(); b1n=b1o.copy(); W2=W2o.copy(); b2n=b2o.copy(); z0=np.zeros(192)
W1[aI]=np.r_[wI[:192],z0,MI*GATE['I'][0]+wI[192:]]; b1n[aI]=MI*(GATE['I'][1]-midI)+bI0
W1[bIu]=np.r_[z0,z0,MI*GATE['I'][0]]; b1n[bIu]=MI*(GATE['I'][1]-midI)
W1[aN]=np.r_[z0,z0,MN*GATE['N'][0]+wN]; b1n[aN]=MN*(GATE['N'][1]-midN)+bN0
W1[bNu]=np.r_[z0,z0,MN*GATE['N'][0]]; b1n[bNu]=MN*(GATE['N'][1]-midN)
b2n+=W2o[:,U4]@am; W2[:,U4]=0
m18,m28=zz_[okA,18].mean(),zz_[okA,28].mean()
for dd,(pa,pb_) in [(18,(aI,bIu)),(28,(aN,bNu))]: W2[dd,:]=0; b2n[dd]=0; W2[dd,pa]=1; W2[dd,pb_]=-1
gI_,gN_=2.0,2.5; Wz=Wzo.copy(); bz=bzo+Wzo[:,18]*m18+Wzo[:,28]*m28; Wz[:,18]=gI_*vI; Wz[:,28]=gN_*vN
def znew(X_): return (np.asarray(Fnn_.gelu(torch.as_tensor(X_@W1.T+b1n)))@W2.T+b2n)[:,:32]
Zn=znew(np.asarray(x,dtype=np.float64)); oth=[i for i in range(32) if i not in (18,28)]
print('other dims maxdiff',np.abs(Zn[okA][:,oth]-zz_[okA][:,oth]).max().round(5),'z18 off max',np.abs(Zn[okA&~hi_,18]).max().round(4),'z28 off max',np.abs(Zn[okA&~hn_,28]).max().round(4))
print('z18 vs dev at ins (f=1): corr',np.corrcoef(Zn[kI,18],rI[kI]-pb[kI])[0,1].round(3),'mean',Zn[kI,18].mean().round(3),'| z28 vs rN corr',np.corrcoef(Zn[kN,28],rN[kN])[0,1].round(3))
ZnI=znew(np.c_[ctxA[cI],np.zeros((len(cI),192)),UI]); print('aug z18 vs target slope',np.polyfit(yI,ZnI[:,18],1).round(3))
tp=type(load('z_in.bias')); cv_=(lambda a: torch.as_tensor(a,dtype=torch.float32)) if tp is torch.Tensor else (lambda a: np.asarray(a,dtype=np.float32))
for nm,a in [('z_post.0.weight',W1),('z_post.0.bias',b1n),('z_post.2.weight',W2),('z_post.2.bias',b2n),('z_in.weight',Wz),('z_in.bias',bz)]: save_edit(nm,cv_(a))
np.savez('/work/edit_parts.npz',vI=vI,vN=vN,W1=W1,b1=b1n,W2=W2,b2=b2n,Wzo=Wzo,bz=bz,m18=m18,m28=m28)
print('edited',edited())
```