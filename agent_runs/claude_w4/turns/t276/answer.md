Every prelude input token at drug bundles is reproducible exactly from data I hold:

- slot token = decayed previous state + `core.slots` (norm ~721, error 8e‑5);
- z token = z_in(posterior mean), exact;
- e token = my stored per‑bundle encoding, exact at drug bundles but approximate at measurement bundles;
- the u token, norm 11.7.

So I can compute E at every bundle. Non‑drug bundles use the constant no‑action u token, LN(mlp2(size.bias)).

Next:
1. Build tokens and E for all 5218 bundles.
2. Compute the injection Jacobian J = ∂E/∂(slot‑residual injection) at drug bundles.
3. Solve the generalized eigenproblem: maximise the signal J·Jᵀ along w relative to the natural second moment of w·E across all bundles and slots.

The top solution gives the E‑direction w that best separates injected signal from natural noise, along with the achievable SNR per unit injection norm. A clean direction needs SNR ≳ 20 at a moderate injection (|Δx| ~10–30, a few % of the residual). That would make the b_in amplifier viable; a much lower value means I need a gated (FFN) amplifier instead.

```python
ew={k:Wt('encode_action.'+k) for k in ['size.bias','mlp2.0.weight','mlp2.0.bias','mlp2.2.weight','mlp2.2.bias','norm.weight','norm.bias']}
uc=LNn(G(ew['size.bias']@ew['mlp2.0.weight'].T+ew['mlp2.0.bias'])@ew['mlp2.2.weight'].T+ew['mlp2.2.bias'],ew['norm.weight'],ew['norm.bias'])
print('|uc| %.2f'%np.linalg.norm(uc))
rows=np.where(~first)[0]; isd=np.zeros(len(sid),bool); isd[dix]=True
dtr=(hr[rows]-hr[rows-1])*3600.
TA=np.zeros((len(rows),7,192))
TA[:,:4]=xo[rows-1,192:960].reshape(-1,4,192).astype(np.float64)*np.exp(-dtr[:,None,None]*np.exp(fa_))+SL
TA[:,4]=ea[rows]; TA[:,5]=uc; TA[:,6]=xo[rows,960:992].astype(np.float64)@zw.T+zb
rpos={r:i for i,r in enumerate(rows)}; di_=np.array([rpos[r] for r in dix]); TA[di_]=TOKa
Eall,_=prelude_np(TA); print('E all',Eall.shape,' check drug rows err %.2e'%np.abs(Eall[di_]-Ep).max())
ismeas=~isd[rows]
Ef=Eall.reshape(-1,192); M2=Ef.T@Ef/len(Ef)
Em_=Eall[ismeas].reshape(-1,192); Ed_=Eall[~ismeas].reshape(-1,192)
J=np.zeros((192,192)); 
for j in range(192):
    dX=np.zeros(192); dX[j]=1.0; Ej,_=prelude_np(TOKa,dX=dX); J[:,j]=(Ej-Ep).reshape(-1,192).mean(0)
from scipy.linalg import eigh
ev_,V_=eigh(J@J.T,M2+1e-12*np.eye(192)); o_=np.argsort(ev_)[::-1]
for i in o_[:6]:
    w=V_[:,i]/np.linalg.norm(V_[:,i]); sig=np.linalg.norm(J.T@w); 
    print('gen-eig %.3g | signal/unit|dx| %.4f | natural rms all %.5f drug %.5f meas %.5f | mean meas %.5f drug %.5f'%(ev_[i],sig,np.sqrt(w@M2@w),np.sqrt(((Ed_@w)**2).mean()),np.sqrt(((Em_@w)**2).mean()),(Em_@w).mean(),(Ed_@w).mean()))
WG=V_[:,o_[:12]]; print('M2 eig smallest',np.round(np.sort(np.linalg.eigvalsh(M2))[:6],6))
```