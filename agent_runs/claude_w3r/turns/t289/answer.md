More units fix the insulin decoder. With 40 random GELU units on its own two reads, insulin residual drops to 0.30 (≈0.2 in the main range, biases ≈0 except +0.3 at rate 0.25). Norepinephrine improves to 0.26 (bias −0.4 at rate 2). And I can afford 40 units per drug: about 150 prelude-FFN units have output impact below 0.074, negligible against slot dims of sd ≈50.

This run builds stages 1–2 numerically:
- stage-1 amplitudes for the rate reads raised to 0.03 to cut decode noise;
- refitted norm3-space decoders;
- 80 lowest-impact units repurposed as two gated 40-unit decoders (λ=15), each writing into a quiet E direction.

I verify that the E-channel is ≈0 at u-absent bundles and tracks the ideal decoded rate at action bundles.

```python
amp=np.array([.05,.03,.03,.03,.03]); D1=np.stack([amp[f]*sig0*U3[:,1+f]/g3n for f in range(5)],1)
Vl0=np.einsum('bqd,df->bqf',MIX,RW2[:192]); XA2=XA+np.einsum('bqf,df->bqd',Vl0,D1)
N3b=LNd(torch.tensor(XA2),g3t,b3t).numpy(); N3h=np.concatenate([N3b,np.ones(N3b.shape[:2]+(1,))],-1).reshape(-1,193)
RHO=np.linalg.lstsq(N3h,Vl0.reshape(-1,5),rcond=None)[0]; ub=np.repeat(upres,4)
print('stage1 decode noise (u present)',np.round((N3h@RHO-Vl0.reshape(-1,5))[ub].std(0),4))
UNITS=np.argsort(imp)[:80]; LAM=15.; DEC={}
for i,(nm,cc,y) in enumerate([('ins',[1,2],yi),('ne',[3,4],yn)]):
    best=min([(float(np.std(rf_fit(cc,y,40,lam=LAM,seed=s)[4]-y)),s) for s in range(8)]); C,b,o_,lam,pr=rf_fit(cc,y,40,lam=LAM,seed=best[1])
    DEC[nm]=dict(cols=cc,C=C,b=b,o=o_,units=UNITS[40*i:40*i+40]); print(nm,'dec res',round(best[0],3),summ(pr,y))
W1n=F1w.numpy().copy(); b1x=F1b.numpy().copy(); W2n=F2w.numpy().copy()
for nm,dd in DEC.items():
    for j,uj in enumerate(dd['units']):
        W1n[uj]=LAM*RHO[:192,0]+sum(dd['C'][k,j]*RHO[:192,f] for k,f in enumerate(dd['cols']))
        b1x[uj]=LAM*(RHO[192,0]-1)+sum(dd['C'][k,j]*(RHO[192,f]+RW2[192,f]) for k,f in enumerate(dd['cols']))+dd['b'][j]
        W2n[:,uj]=0
def prelude_out(W2_):
    pre=N3b@W1n.T+b1x; xf=XA2+Fnn.gelu(torch.tensor(pre)).numpy()@W2_.T+F2b.numpy(); return LNd(torch.tensor(xf),gpt,bpt).numpy(), Fnn.gelu(torch.tensor(pre)).numpy()
E0,Hu=prelude_out(W2n); evE,QE=np.linalg.eigh(np.cov(E0.reshape(-1,192).T)); print('E sd bottom',np.round(np.sqrt(np.maximum(evE[:4],0)),5))
GE=.05; gpn=gpt.numpy(); DOUT={'ins':GE*sig0*QE[:,1]/gpn,'ne':GE*sig0*QE[:,2]/gpn}
for nm,dd in DEC.items(): W2n[:,dd['units']]=np.outer(DOUT[nm],dd['o'])
E1,Hu=prelude_out(W2n); Eh=np.concatenate([E1,np.ones(E1.shape[:2]+(1,))],-1).reshape(-1,193)
Vt=np.concatenate([Vl0,np.ones(Vl0.shape[:2]+(1,))],-1).reshape(-1,6)
for nm,dd in DEC.items():
    cc=dd['cols']; Rt=Vt[:,cc]+RW2[192,cc]; ideal=Fnn.gelu(torch.tensor(LAM*(Vt[:,[0]]-1)+Rt@dd['C']+dd['b'])).numpy()@dd['o']
    actual=Hu.reshape(-1,768)[:,dd['units']]@dd['o']; vr=np.linalg.lstsq(Eh,actual,rcond=None)[0]; rd=Eh@vr
    print(nm,'FFN-out vs ideal noise (u pres)',round(float((actual-ideal)[ub].std()),3),'| E-read noise',round(float((rd-actual).std()),4),'| u-absent: actual max',round(float(np.abs(actual[~ub]).max()),4),' read max',round(float(np.abs(rd[~ub]).max()),3),'| action range',np.round([actual[ub].min(),actual[ub].max()],2))
    dd['vE']=vr
print('side effect |E1-E_orig| u-absent mean',round(float(np.linalg.norm(E1-np.array(EE).reshape(E1.shape),axis=-1)[~upres].mean()),4))
```