The gate failures come from bundle k=0, the first bundle of each stay. There the state is initial, head 2 attends the z token (0.40) and the slots, and the gate read is 0.9. Those tokens were never in my negative set. At 125 no-action bundles head 2 attends the empty u token (u0), which is harmless. Only 22 of 1446 drug bundles split their attention, and those will simply get no effect.

Fix: refit the gate read with the actual attention-weighted mixtures at u-absent bundles (including k=0) as negatives. Add the same constraint to the rate reads. Then rebuild 8-unit decoders per drug. Success means decoder output ≈0 at all u-absent local bundles, with similar accuracy at action bundles.

```python
MIX=np.einsum('bqj,bjd->bqd',ATT2,N1T); Mneg=MIX[~upres].reshape(-1,192)
def fitread(Xp,yp,negs,wn,lam=1e-6,icpt=False):
    Xp_=np.hstack([Xp,np.ones((len(Xp),1))]) if icpt else Xp
    A=Xp_.T@Xp_/len(Xp_)+lam*np.eye(Xp_.shape[1])
    for Xn,w in zip(negs,wn):
        Xn_=np.hstack([Xn,np.ones((len(Xn),1))]) if icpt else Xn; A+=w*Xn_.T@Xn_/len(Xn_)
    return np.linalg.solve(A,Xp_.T@yp/len(Xp_))
w_g2=fitread(N1X,np.ones(len(N1X)),[NEG,Mneg,b1n[None]],[1.,3.,1.])
print('gate2: pos res',round(float(np.std(N1X@w_g2-1)),4),'NEG max',round(float(np.abs(NEG@w_g2).max()),3),'mix-neg max',round(float(np.abs(Mneg@w_g2).max()),3),'u0',round(float(b1n@w_g2),4))
cols=[]; 
for y in [yi,yn]:
    T=np.stack([y,np.sqrt(y)],1); W_=fitread(N1X,T,[NEG,Mneg],[.1,.3],icpt=True); cols+=[W_[:,0],W_[:,1]]
RW2=np.stack([np.r_[w_g2,0]]+cols,1); Vu2=np.hstack([N1X,np.ones((len(N1X),1))])@RW2
MIXh=np.concatenate([MIX,np.ones(MIX.shape[:2]+(1,))],-1); Vl=np.einsum('bqd,df->bqf',MIXh,RW2)
print('local V no-action max|.|',np.round(np.abs(Vl[~upres]).reshape(-1,5).max(0),2))
def build_dec2(cols,y,K=8,lam=40.,seed=1):
    rng=np.random.default_rng(seed); R=Vu2[:,cols]; m=y>0
    C=rng.normal(size=(len(cols),K))/R[m].std(0)[:,None]; b=-(R[m].mean(0)@C)+rng.normal(size=K)+1.0
    H=Fnn.gelu(torch.tensor(lam*(Vu2[:,[0]]-1)+R@C+b)).numpy(); o=np.linalg.lstsq(H,y,rcond=None)[0]; return C,b,o,lam,H@o
DECS={}
for nm,cc,y in [('ins',[1,2],yi),('ne',[3,4],yn)]:
    best=min([(np.std(build_dec2(cc,y,seed=s)[4]-y),s) for s in range(10)]); C,b,o,lam,pr=build_dec2(cc,y,seed=best[1]); DECS[nm]=(cc,C,b,o,lam)
    rs=[(round(float(np.std(pr[s_]-y[s_])),2),round(float(np.mean(pr[s_]-y[s_])),2)) for lo,hi in [(0,1e-9),(1e-9,.3),(.3,1),(1,2),(2,5),(5,30)] for s_ in [((y>=lo)&(y<hi)) if hi>1e-9 else (y==0)] if s_.sum()>5]
    Vq=Vl.reshape(-1,5); out=Fnn.gelu(torch.tensor(lam*(Vq[:,[0]]-1)+Vq[:,cc]@C+b)).numpy()@o
    print(nm,'(res,bias) by range',rs,'| local u-absent out max',round(float(np.abs(out[np.repeat(~upres,4)]).max()),4))
```