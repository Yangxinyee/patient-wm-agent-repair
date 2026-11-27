The attention pattern makes the plan workable.
- **Head roles:** head 1 attends 100% to the e token (readings); heads 0 and 2 attend almost entirely to the drug token u; head 3 splits between u (0.69 at insulin bundles) and the z token (0.26 at insulin, 0.68 at NE bundles). So z can reach the state, but only through head 3.
- **Quiet directions:** there are directions in normalised-token space along which every token type varies by only 0.001–0.006 (eigenvalues ≈ 0). A large rank-1 gain on head 3's value projection along such a direction therefore has negligible side effects, about 1% of the signal by my estimate.

The edit:
1. **Gated z-posterior units** (saved earlier) write the insulin deviation (rate − policy-expected rate) into z dim 18 and the NE rate into z dim 28.
2. **z_in columns 18 and 28** move the z token along two orthogonal quiet directions, 0.15 normalised units per unit signal.
3. **Head-3 value rows** get rank-1 terms mapping those directions to the model's own dose-response direction. That direction is the residual change the slots receive when the u token's rate rises, projected onto head 3's output space, scaled to κ times the model's natural per-unit-rate response. I start at κ = 6 for insulin and 10 for NE, then calibrate with dose_edit.

Locally, the z token should move by ≈0.15·dev at insulin bundles and stay unchanged elsewhere.

```python
P_=np.load('/work/edit_parts.npz'); fI_d=Pc@EV[:,0]; fI_d/=np.linalg.norm(fI_d); fN_d=Pc@EV[:,1]; fN_d-=fI_d*(fI_d@fN_d); fN_d/=np.linalg.norm(fN_d)
Wo=np.asarray(load_original('core.prelude.0.attn.out.weight'),dtype=np.float64); Wo3=Wo[:,144:192]
def rrate(Uaug,kk,rr):
    i1=np.arange(len(kk))*4+1; i2=i1+1; L1=g1*(Uaug[i1]-Uaug[i1].mean(1,keepdims=True))/np.sqrt(Uaug[i1].var(1,keepdims=True)+1e-5)+be1
    L2=g1*(Uaug[i2]-Uaug[i2].mean(1,keepdims=True))/np.sqrt(Uaug[i2].var(1,keepdims=True)+1e-5)+be1
    dV=((L2-L1)@Wv.T)/rr[kk][:,None]; au=AT[kk][:,:,:,5].mean(2)
    return np.einsum('nh,nhd->nd',au,np.stack([dV[:,48*h:48*h+48]@Wo[:,48*h:48*h+48].T for h in range(4)],1)).mean(0)
rI_=rrate(UI,kI,rI); rN_=rrate(UN,kN,rN)
a3I=np.linalg.lstsq(Wo3,rI_,rcond=None)[0]; a3N=np.linalg.lstsq(Wo3,rN_,rcond=None)[0]
print('|r_rate| I,N',np.linalg.norm(rI_).round(3),np.linalg.norm(rN_).round(3),'captured rho',(np.linalg.norm(Wo3@a3I)/np.linalg.norm(rI_)).round(3),(np.linalg.norm(Wo3@a3N)/np.linalg.norm(rN_)).round(3))
azI=AT[hi_&okA][:,3,:,6].mean(); azN=AT[hn_&okA][:,3,:,6].mean(); c_sh=0.15; sz=sdv[okA,6,0].mean()
def build(kI_,kN_):
    gI=kI_/(azI*c_sh); gN=kN_/(azN*c_sh)
    Wz=P_['Wzo'].copy(); Wz[:,18]=c_sh*sz*fI_d; Wz[:,28]=c_sh*sz*fN_d
    Wkv2=Wkv.copy(); Wkv2[192+144:]+=gI*np.outer(a3I,fI_d/g1)+gN*np.outer(a3N,fN_d/g1)
    return Wz,Wkv2,gI,gN
kapI,kapN=6.0,10.0; Wz,Wkv2,gI,gN=build(kapI,kapN)
W1,b1n,W2,b2n,bz=P_['W1'],P_['b1'],P_['W2'],P_['b2'],P_['bz']
Zn=(np.asarray(Fnn_.gelu(torch.as_tensor(np.asarray(x,dtype=np.float64)@W1.T+b1n)))@W2.T+b2n)[:,:32]; Ztn=Zn@Wz.T+bz
xz=(Ztn-Ztn.mean(1,keepdims=True))/np.sqrt(Ztn.var(1,keepdims=True)+1e-5)
print('z fI proj: ins corr w dev',np.corrcoef(xz[kI]@fI_d,rI[kI]-pb[kI])[0,1].round(3),'slope',np.polyfit(rI[kI]-pb[kI],xz[kI]@fI_d,1).round(3),'nondrug sd',(xz[okA&~hi_&~hn_]@fI_d).std().round(4),'| NE proj slope',np.polyfit(rN[kN],xz[kN]@fN_d,1).round(3))
print('gI,gN',round(gI,1),round(gN,1),'side sd (slots,e,u) on fI',[float((XH[okA][:,j].reshape(-1,192)@fI_d).std().round(5)) for j in (slice(0,4),4,5)])
for nm,a in [('z_post.0.weight',W1),('z_post.0.bias',b1n),('z_post.2.weight',W2),('z_post.2.bias',b2n),('z_in.weight',Wz),('z_in.bias',bz),('core.prelude.0.attn.kv.weight',Wkv2)]: save_edit(nm,cv_(a))
print('edited',edited())
```