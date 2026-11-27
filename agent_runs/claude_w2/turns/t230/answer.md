Run 224 found the carrier. In `core.prelude.0`, attention heads 0 and 2 send all 4 slots to the action token u with weight 1.000 ± 0.000 at every insulin and norepinephrine bundle. Their output is therefore the value vector of u. A linear readout of head 2's output gives the insulin rate with R² 0.980 and the norepinephrine rate with R² 0.951 (head 0: 0.986 / 0.963).

So the dose enters the state cleanly. It is diluted afterwards by the LayerNorms, the memory blocks and the coda, which is why the head sees only a weak, noisy version.

The edit therefore becomes a rank-2 update to `core.prelude.0.attn.out.weight[:, 96:144]` (head 2):
- **Insulin channel:** residual += c_I·(w_I·o₂)·v_I.
- **Norepinephrine channel:** residual += c_N·(w_N·o₂)·v_N.
- **Readouts w_I, w_N:** fitted on drug bundles plus non-drug bundles (target 0) with no intercept, so no-drug bundles inject nothing.
- **Directions v_I, v_N:** the top two right-singular directions of the linear skip path c_out·diag(softplus δ)·b_in·diag(γ_prelude_norm), mean-removed, so the signal survives into `out`.
- **Amplitude:** κ = 0.1 of the slot-residual norm per typical dose (2 U/h insulin, 0.4 mcg/kg/min norepinephrine).

In the same run I re-collect the 4-arm teacher-forced data with this edited core and measure two things:
1. **Side effect:** the factual value-NLL change from the core injection alone.
2. **Gain:** whether the head-input decoders of d1 and n1 now reach a causal slope ≈ 1 with small factual error.

**Confirm:** slope above 0.85, error sd well under the signal sd, and an NLL change under about 0.002.

```python
o2=o[:,2,:4].mean(1).numpy().astype(np.float64); dr=(RI>0)|(RN>0)
wI48=np.linalg.lstsq(o2,RI,rcond=None)[0]; wN48=np.linalg.lstsq(o2,RN,rcond=None)[0]
for nm,w_,y_ in [('I',wI48,RI),('N',wN48,RN)]:
    p_=o2@w_; print(nm,'R2 %.4f resid sd %.3f | nodrug readout mean %.3f sd %.3f'%(1-np.var(y_-p_)/np.var(y_),np.std(y_-p_),p_[~dr].mean(),p_[~dr].std()))
gp=W['core.prelude_norm.weight'].double().numpy(); sp_=F.softplus(W['core.delta_in']).double().numpy()
Mlin=W['core.c_out.weight'].double().numpy()@np.diag(sp_)@W['core.b_in.weight'].double().numpy()@np.diag(gp)
Uu,Ss,Vt=np.linalg.svd(Mlin); print('sing vals',np.round(Ss[:5],3))
vI=Vt[0]-Vt[0].mean(); vI/=np.linalg.norm(vI); vN=Vt[1]-Vt[1].mean(); vN-=vI*(vI@vN); vN/=np.linalg.norm(vN)
Xo=layer_sa(X7,W,'core.prelude.0'); sig=Xo[:,:4].std(-1).median().item(); nsc=sig*math.sqrt(192); print('slot resid sigma',sig,'norm',nsc)
def make_core_chan(W_,kI,kN):
    W2_=dict(W_); ow=W_['core.prelude.0.attn.out.weight'].clone().double()
    ow[:,96:144]+=kI*nsc/2.0*np.outer(vI,wI48)+kN*nsc/0.4*np.outer(vN,wN48)
    W2_['core.prelude.0.attn.out.weight']=ow.float(); return W2_
np.savez('/work/corechan.npz',wI48=wI48,wN48=wN48,vI=vI,vN=vN,nsc=nsc)
W2=make_core_chan(W,0.1,0.1)
t0=time.time(); RT2,resT2,DT2=cf_collect(tr,idsT,armsT,W2)
obT=obs_bins(DT); ob2=obs_bins(DT2); f0=DT['arm']==0
n0=nll_of(DT['P'][f0],obT[f0]); n2=nll_of(DT2['P'][f0],ob2[f0])
for c in MEAS:
    mc_=DT['code'][f0]==CODE[c]; print(c,'NLL orig %.4f core-edit %.4f'%(np.mean(n0[mc_]),np.mean(n2[mc_])))
print('all NLL orig %.5f edit %.5f'%(np.mean(n0),np.mean(n2)))
st0=torch.zeros(len(DT2['code']),dtype=torch.long); cT=torch.as_tensor(DT2['code'],dtype=torch.long)
pr2=mlp(DT2['base'],W2,'z_prior')
_,w20=head_value_logits(DT2['base'],cT,DT2['f8'],DT2['lb'],st0,W2,DT2['zmu'],1)
_,w21=head_value_logits(DT2['base'],cT,DT2['f8'],DT2['lb'],st0,W2,DT2['zmu']+torch.exp(pr2[:,32:])*torch.randn(DT2['zmu'].shape,generator=torch.Generator().manual_seed(7)),1)
X0,X1=w20.numpy().astype(np.float64),w21.numpy().astype(np.float64)
_=stacked(glc,dI[:,0],[0,1,10,100],1.0,'CORE-EDIT INS d1'); _=stacked(mapr,dN[:,0],[0,1,10,100],1.0,'CORE-EDIT NE n1')
print('time',time.time()-t0)
```