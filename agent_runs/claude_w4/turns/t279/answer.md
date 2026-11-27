The pieces fit together:

- **Coda:** my coda replica inverts exactly. Pushing a state change through c_out⁻¹ and the coda reproduces the target (cosine 0.996–0.999, ratio 0.99–1.02), so Δy ≈ c_out·Δh_new holds well.
- **Insulin target direction (dF):** in the span of natural directions, a state change exists that moves finger and lab glucose without touching MAP.
- **NE target direction (dM):** a matching change moves arterial and cuff MAP equally without touching glucose.
- **Cost of equal finger and lab effects:** forcing them equal more than doubles the size of the state change (6.7 versus 2.95 for a near‑nuG direction). So I'll target lab at 0.8 of finger, the model's natural ratio, to keep the perturbation small.

Now I'll build the carrier half of the edit (step 1 of 2) and verify it on CPU:
1. **Rate readouts on the prelude's normalised u token.** They are fitted to give (rate − θ·mean rate), centring for factual fit with θ=1 to start, and exactly 0 for the no‑action token and for the other drug. Training data include combos and the ×0, ×0.5, ×1.5, ×2 and ×3 edited rates.
2. **Head‑1 value directions** orthogonal to all original u values, so nothing leaks at measurement bundles.
3. **Residual injection directions** that write a signal s = 0.05 per rate unit onto the two silent E‑directions w_I and w_N.

The check is that the replica E shows w·E ≈ s·f at drug bundles, including doubled rates, and ≈ 0 (rms ≪ s) at every non‑drug bundle.

```python
LN1=lambda x: LNn(x,PW['core.prelude.0.norm1.weight'],PW['core.prelude.0.norm1.bias'])
rbI=float(np.nanmean(rI_d)); rbN=float(np.nanmean(rN_d)); TH=1.0; print('mean rates I %.3f N %.3f'%(rbI,rbN), SHAPES['core.delta_in'],SHAPES['core.b_in.weight'])
X_,Y_,C_=[],[],[]
def add(rI,rN,wt):
    rI=np.asarray(rI,float); rN=np.asarray(rN,float); X_.append(LN1(u_of(rI,rN)))
    Y_.append(np.stack([np.where(np.isfinite(rI),np.nan_to_num(rI)-TH*rbI,0.),np.where(np.isfinite(rN),np.nan_to_num(rN)-TH*rbN,0.)],1)); C_.append(np.full(len(rI),wt))
gI=np.linspace(0,11.5,117); gN=np.linspace(0,6.4,129); nan=np.nan
add(gI,np.full_like(gI,nan),3.); add(np.full_like(gN,nan),gN,3.)
cI,cN=np.meshgrid(np.linspace(0,11.5,24),np.linspace(0,6.4,24)); add(cI.ravel(),cN.ravel(),1.)
for m in [1.,0.,.5,1.5,2.,3.]: add(rI_d*m,rN_d,10. if m==1 else 3.); add(rI_d,rN_d*m,3.)
X_=np.vstack(X_); Y_=np.vstack(Y_); C_=np.concatenate(C_)
X_=np.vstack([X_,LN1(uc)[None]]); Y_=np.vstack([Y_,[[0.,0.]]]); C_=np.r_[C_,1e5]
Xw=X_*np.sqrt(C_)[:,None]; RO=np.linalg.solve(Xw.T@Xw+1e-3*np.eye(192),Xw.T@(Y_*np.sqrt(C_)[:,None]))   # 192x2
res_=X_@RO-Y_; print('readout rms all I %.3f N %.3f | at u_const %.2e %.2e'%(np.sqrt((res_[:,0]**2).mean()),np.sqrt((res_[:,1]**2).mean()),*(LN1(uc)@RO)))
fd=LN1(TOKa[:,5])@RO; tgtI=np.where(np.isfinite(rI_d),np.nan_to_num(rI_d)-TH*rbI,0); tgtN=np.where(np.isfinite(rN_d),np.nan_to_num(rN_d)-TH*rbN,0)
print('drug bundles readout rms I %.3f N %.3f'%(np.sqrt(((fd[:,0]-tgtI)**2).mean()),np.sqrt(((fd[:,1]-tgtN)**2).mean())))
kvO=PW['core.prelude.0.attn.kv.weight']; woO=PW['core.prelude.0.attn.out.weight']; V1_=kvO[240:288]
vals=np.vstack([X_,LN1(TOKa[:,4]),LN1(TOKa[:,6])])@V1_.T; _,svv,Vtv=np.linalg.svd(vals,full_matrices=True); aI=Vtv[-1]; aN=Vtv[-2]
print('value sv tail',np.round(svv[-4:],4),' max |a.val| I %.4f N %.4f'%(np.abs(vals@aI).max(),np.abs(vals@aN).max()))
wI=V_[:,o_[0]]/np.linalg.norm(V_[:,o_[0]]); wN=V_[:,o_[1]]/np.linalg.norm(V_[:,o_[1]]); W2=np.stack([wI,wN],1); s_sig=0.05
T2=J.T@W2@np.linalg.inv(W2.T@J@J.T@W2)*s_sig; print('|tI| %.2f |tN| %.2f per unit'%tuple(np.linalg.norm(T2,axis=0)))
PWe=dict(PW); kvE=kvO.copy(); kvE[240:288]+=np.outer(aI,RO[:,0])+np.outer(aN,RO[:,1]); woE=woO.copy(); woE[:,48:96]+=np.outer(T2[:,0],aI)+np.outer(T2[:,1],aN)
PWe['core.prelude.0.attn.kv.weight']=kvE; PWe['core.prelude.0.attn.out.weight']=woE
Ee,_=prelude_np(TA,W=PWe); sg=Ee@W2/s_sig   # (rows,4,2)
md=~ismeas
print('drug: sigI vs tgt rms %.3f corr %.4f | sigN rms %.3f corr %.4f'%(np.sqrt(((sg[md,:,0].mean(1)-tgtI)**2).mean()),np.corrcoef(sg[md,:,0].mean(1),tgtI)[0,1],np.sqrt(((sg[md,:,1].mean(1)-tgtN)**2).mean()),np.corrcoef(sg[md,:,1].mean(1),tgtN)[0,1]))
print('meas bundles: sig rms I %.4f N %.4f mean %.4f %.4f | E change rel %.2e'%(np.sqrt((sg[ismeas,:,0]**2).mean()),np.sqrt((sg[ismeas,:,1]**2).mean()),sg[ismeas,:,0].mean(),sg[ismeas,:,1].mean(),np.linalg.norm(Ee[ismeas]-Eall[ismeas])/np.linalg.norm(Eall[ismeas])))
T2x=TOKa.copy(); T2x[:,5]=u_of(rI_d*2,rN_d); Ex,_=prelude_np(T2x,W=PWe); sx=(Ex@W2/s_sig).mean(1)
print('x2 insulin: sigI - base vs rI: slope %.3f rms %.3f'%(np.polyfit(np.nan_to_num(rI_d),sx[:,0]-sg[md,:,0].mean(1),1)[0],np.sqrt(((sx[:,0]-sg[md,:,0].mean(1)-np.nan_to_num(rI_d))**2).mean())))
```