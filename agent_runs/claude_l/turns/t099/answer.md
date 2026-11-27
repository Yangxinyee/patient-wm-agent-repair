Bundle times and atom counts now match the tool for all 100 stays. Both admission rows contribute an END atom at the stay's end.

Next I return to the z mismatch at k=1, where the context is the same for every stay. I fit the unknown context offset again, now with an analytic Jacobian so it runs quickly. Then I look at the residual stay by stay, against which atoms the k=1 bundle contains.

- If the residual is near zero for bundles without glucose_finger atoms and large only for those with them, the problem is that atom's encoding (for example its bins), as run 66 hinted.
- If the residual is spread evenly, the encoder structure or the z_post input layout is wrong.

```python
BUN={s:build_bundles(tu[tu.stay_id==s]) for s in stays}
from scipy.optimize import least_squares
from scipy.special import erf
def act_np(name,x):
    if name=='silu': return x/(1+np.exp(-x))
    if name=='gelu': return 0.5*x*(1+erf(x/np.sqrt(2)))
    return np.maximum(x,0)
def dact_np(name,x):
    if name=='silu': s=1/(1+np.exp(-x)); return s*(1+x*(1-s))
    if name=='gelu': return 0.5*(1+erf(x/np.sqrt(2)))+x*np.exp(-x*x/2)/np.sqrt(2*np.pi)
    return (x>0).astype(float)
def enc_bundle(atoms):
    meas=[a for a in atoms if not a['act']]; act=[a for a in atoms if a['act']]
    e=set_enc(W0,meas,'encode') if meas else torch.zeros(192,dtype=T)
    u=set_enc(W0,act,'encode_action') if act else torch.zeros(192,dtype=T)
    return e,u
W1=W0['z_post.0.weight'].numpy(); b1=W0['z_post.0.bias'].numpy(); W2=W0['z_post.2.weight'].numpy()[:32]; b2=W0['z_post.2.bias'].numpy()[:32]
k1=[s for s in stays if abs(BUN[s][1][0]-BUN[s][0][0]-1.0)<1e-6]
Zi=np.stack([X[row_of[s][1],960:] for s in k1])
def fit_a(ea,za,blk=1):
    ACTMAP.clear(); ACTMAP.update({'encode.mlp1':ea,'encode.mlp2':ea,'encode_action.mlp1':ea,'encode_action.mlp2':ea})
    Ev=np.stack([enc_bundle(BUN[s][1][1])[0].numpy() for s in k1])
    Vv=Ev@W1[:,192*blk:192*(blk+1)].T+b1
    fun=lambda a: ((act_np(za,a[None,:]+Vv)@W2.T+b2)-Zi).ravel()
    jac=lambda a: np.concatenate([W2*dact_np(za,a+Vv[i])[None,:] for i in range(len(k1))],0)
    r=least_squares(fun,np.zeros(192),jac=jac,method='trf',max_nfev=200)
    return r
import time
for ea,za,blk in [('gelu','silu',1),('silu','silu',1),('gelu','gelu',1),('gelu','silu',2),('gelu','silu',0)]:
    t0=time.time(); r=fit_a(ea,za,blk); res_=r.fun.reshape(len(k1),32)
    print(ea,za,blk,'rms %.4f'%np.sqrt(np.mean(res_**2)),'time %.0fs'%(time.time()-t0))
r=fit_a('gelu','silu',1); res_=np.abs(r.fun.reshape(len(k1),32)).max(1)
for i in np.argsort(res_)[[0,1,2,3,40,80,-3,-2,-1]]:
    print(k1[i], round(res_[i],4), [(a['code'],a['vb']) for a in BUN[k1[i]][1][1]])
```