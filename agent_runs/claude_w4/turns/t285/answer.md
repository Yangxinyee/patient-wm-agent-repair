The step structure explains why v3 fails. Each update keeps only about 37% of the old h (a_bar 0.30–0.46). `softplus(delta_in)` ≈ 1, so `b_in(E)` is the main way E enters h. Slot norms are 560–1100 while `core.slots` are under 1, so the slot tokens are essentially the decayed h. The 1/σ of `prelude_norm` (σ ≈ 57) shrinks any head-1 injection about 60-fold.

Plan for the dedicated channel:
- Head 1 adds a modest c·rate along a slot-token direction q, small enough not to distort the normalisations.
- A rank-1 term added to `core.b_in` reads that direction of E and writes the target change into h_new.
- The target is the model's own glucose-news direction (or MAP for NE), mapped back through `c_out`.

This requires three things:
1. Directions where the normalised slot tokens hardly vary naturally, so the added `b_in` term picks up little noise.
2. The news directions nuG and nuM: regress the slot change at reading bundles on the factual reading surprises from fwd_2.
3. The conditioning of `c_out` and `b_in`.

I'll compute all three from the fwd_3 states and save them to disk. The channel is workable if the smallest natural-variance eigenvalues of LN(slot token) are far below about 0.03 (≈ (c/σ)²), the news directions agree across slots, and `c_out` is well conditioned.

```python
L = lambda n: np.asarray(load_original(n), dtype=np.float64)
def ln(x, w=None, b=None, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x-m)**2).mean(-1, keepdims=True); y = (x-m)/np.sqrt(v+eps)
    return y if w is None else y*w + b
fa = L('core.flow.a'); slots = L('core.slots')
X = s3['x'].astype(np.float64); sid=s3['stay_id']; hr=s3['hour']; H = X[:,192:960].reshape(-1,4,192)
prev_ok = np.r_[False, sid[1:]==sid[:-1]]; dt = np.r_[0, np.diff(hr)]*3600; idx = np.where(prev_ok)[0]
Xs = H[idx-1]*np.exp(-dt[idx,None,None]*np.exp(fa)[None,None,:]) + slots[None]
sig = Xs.std(-1); print('LN sigma of slot tokens q', np.round(np.quantile(sig,[.01,.1,.5,.9,.99]),1))
N = ln(Xs).reshape(-1,192); C = np.cov(N.T); ev, U = np.linalg.eigh(C)
print('eig of cov(LN slot tok) smallest', np.round(ev[:12],6), 'largest', np.round(ev[-5:],3), 'n_eig<1e-4', (ev<1e-4).sum(), '<1e-3', (ev<1e-3).sum())
for n in ['core.c_out.weight','core.b_in.weight']:
    s = np.linalg.svd(L(n), compute_uv=False); print(n, 'sv', np.round(s[[0,1,5,20,100,180,191]],3))
gp, bp = L('core.prelude_norm.weight'), L('core.prelude_norm.bias'); print('prelude_norm w q', np.round(np.quantile(gp,[0,.1,.5,.9,1]),3), 'b q', np.round(np.quantile(bp,[0,.5,1]),3))
# reading surprises per bundle from fwd_2
f2['hk']=f2.hour.round(4); f2['s']=f2.observed-f2.predicted_mean
S = f2.pivot_table(index=['stay_id','hk'], columns='event', values='s', aggfunc='sum').reset_index()
kk = pd.DataFrame({'stay_id':sid,'hk':np.round(hr,4)}).merge(S, on=['stay_id','hk'], how='left')
ev4 = ['glucose_finger','glucose_lab','map_arterial','map_cuff']; Sm = kk[ev4].fillna(0).values
anyr = (kk[ev4].notna().any(axis=1).values) & ~s3['has_insulin'] & ~s3['has_norepinephrine'] & prev_ok
r = np.where(anyr)[0]; dY = (H[r]-H[r-1]); A = np.c_[np.ones(len(r)), Sm[r]]
B = np.linalg.lstsq(A, dY.reshape(len(r),-1), rcond=None)[0]   # (5, 768)
nu = B[1:].reshape(4,4,192)  # event, slot, dim
for j,e in enumerate(ev4):
    ns = nu[j]; cs = [np.dot(ns[0],ns[s])/np.linalg.norm(ns[0])/np.linalg.norm(ns[s]) for s in range(4)]
    print(e, 'n', (kk[e].notna().values[r]).sum(), 'norm per slot', np.round(np.linalg.norm(ns,axis=1),3), 'cos slot0 vs s', np.round(cs,3))
print('cos finger-lab', np.round(np.dot(nu[0].ravel(),nu[1].ravel())/np.linalg.norm(nu[0])/np.linalg.norm(nu[1]),3), 'art-cuff', np.round(np.dot(nu[2].ravel(),nu[3].ravel())/np.linalg.norm(nu[2])/np.linalg.norm(nu[3]),3), 'finger-art', np.round(np.dot(nu[0].ravel(),nu[2].ravel())/np.linalg.norm(nu[0])/np.linalg.norm(nu[2]),3))
np.savez('/work/an1.npz', nu=nu, ev=ev, U=U, sig=sig)
```