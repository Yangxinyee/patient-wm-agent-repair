`run_stay` and its helpers let me replay the state update locally with edited weights. That makes a local test harness possible.

**Design under test:**
1. **Write:** dead z_post units compute the dose deviation and send it into the state through a repurposed z_in column.
2. **Read:** dead z_prior units read that signal from the context at the next prediction, and z_head turns it into a shift of the reading prediction.

The write side is weak, but it only needs to be clean: easy to tell apart from the context's natural variation. This run measures that:
- the Jacobian of the next-prediction context with respect to the z token, on 40 insulin and 40 norepinephrine drug bundles;
- the best signal-to-noise direction against the natural context covariance, and how consistent its gain is across bundles;
- linearity at larger injections;
- whether z_prior has dead units I can use for the read side.

**Decision rule:** if the signal-to-noise per unit token is large compared with the natural token spread, and the gains are consistent and linear, go ahead. If not, the design is not viable.

```python
import numpy as np, torch, time
t0=time.time()
sid = np.array([d['sid'] for d in INFO]); N=len(sid)
j0 = np.zeros(N,int)
for k in range(1,N): j0[k] = j0[k-1] if sid[k]==sid[k-1] else k
dt_ = WT['core.flow.a'].dtype; fa = WT['core.flow.a']; Vd = {'incl':1,'unit':'s','age':'both'}
X3t = torch.as_tensor(X3, dtype=dt_)
def ctx_after(k, h1, o1):
    dtn = dts[k+1]; return ctx_from(o1, h1*torch.exp(-dtn*torch.exp(fa)), dtn)
def step_k(k):
    hp = X3t[k-1,192:960].reshape(4,192); op = X3t[k-1,:192]; dtk = dts[k]
    hf = hp*torch.exp(-dtk*torch.exp(fa)); ctx = ctx_from(op, hf, dtk)
    z = zpost_mean(ctx, Ee[k], Uu[k])
    idx_m = np.arange(max(j0[k], k+1-128), k+1); ages = (tmin[k]-tmin[idx_m])*60.0
    h1,o1 = core_step(hf, Ee[k], Uu[k], z, Ee[idx_m], ages, Vd)
    return ctx_after(k, h1, o1)
def ctx_nat(k): return ctx_after(k, X3t[k,192:960].reshape(4,192), X3t[k,:192])
ok = np.array([k>j0[k] and k+1<N and sid[k+1]==sid[k] for k in range(N)])
hI_ = np.asarray(hasI).astype(bool); hN_ = np.asarray(hasN).astype(bool)
rng = np.random.default_rng(0)
ta=time.time(); kk0=int(np.where(ok&hI_)[0][0]); c0=step_k(kk0); tstep=time.time()-ta
print('tstep', round(tstep,4), 'relerr vs tool', float((c0-ctx_nat(kk0)).norm()/ctx_nat(kk0).norm()))
nb = int(max(10, min(40, 330/(193*tstep)/2)))
kI_ = rng.choice(np.where(ok & hI_)[0], nb, replace=False); kN_ = rng.choice(np.where(ok & hN_ & ~hI_)[0], nb, replace=False)
zb = WT['z_in.bias']; zb0 = zb.clone(); eps=0.02
def jac(k):
    base = step_k(k); J = torch.zeros(192,192, dtype=dt_)
    for j in range(192):
        zb[j] = zb0[j]+eps; J[:,j] = (step_k(k)-base)/eps; zb[j] = zb0[j]
    return J.numpy()
JI_ = np.stack([jac(int(k)) for k in kI_]); JN_ = np.stack([jac(int(k)) for k in kN_])
kd_ = np.where(ok & (hI_|hN_))[0]
CN = torch.stack([ctx_nat(int(k)) for k in kd_]).numpy(); Sig_ = np.cov(CN.T)
Sinv = np.linalg.inv(Sig_ + 1e-6*np.eye(192)*np.trace(Sig_)/192)
res={}
for nm, Js in [('I',JI_),('N',JN_)]:
    Jb = Js.mean(0); Mx = Jb.T@Sinv@Jb; ev, evec = np.linalg.eigh(Mx); t1 = evec[:,-1]; w = Sinv@Jb@t1
    gi = np.einsum('d,nde,e->n', w, Js, t1)/(w@Jb@t1); res[nm]=(t1,w,Jb)
    print(nm, 'n', len(Js), 'sqrt top eigs', np.sqrt(np.maximum(ev[-4:],0)).round(2), 'gain mean/sd/min', 1.0, gi.std().round(3), gi.min().round(3))
tok = X3[:,960:]@WT['z_in.weight'].numpy().T + zb0.numpy()
print('token norm mean', np.linalg.norm(tok,axis=1).mean().round(2), 'nat token spread', np.sqrt(np.trace(np.cov(tok.T))).round(3))
t1,w,Jb = res['I']; lin=[]
for a in [0.5,2,8]:
    gs=[]
    for k in kI_[:5]:
        k=int(k); base=step_k(k); zb.copy_(zb0 + a*torch.as_tensor(t1,dtype=dt_)); c=step_k(k); zb.copy_(zb0)
        gs.append(float(w@(c-base).numpy())/a/float(w@Jb@t1))
    lin.append(np.round(gs,2).tolist())
print('linearity (gain at a=0.5,2,8):', lin)
Wp, bp = WT['z_prior.0.weight'].numpy(), WT['z_prior.0.bias'].numpy()
hidp = torch.nn.functional.gelu(torch.as_tensor(CN@Wp.T+bp)).numpy()
print('z_prior dead units', int((hidp.std(0)<0.01).sum()), 'time', round(time.time()-t0,1))
```