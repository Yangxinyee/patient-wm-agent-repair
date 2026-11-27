Everything checks out:

- **u replica:** exact (max error 1e‑6). I can now compute the u token for any insulin/NE rate pair, including the ×0.5, ×1.5, ×2 and ×3 edits.
- **Natural glucose‑news direction nuG:** a clean glucose lever. Per unit it moves predicted finger glucose by +0.35 and lab by +0.28, with arterial and cuff MAP essentially unchanged.
- **Natural MAP direction nuM:** not clean. It moves MAP (arterial +0.12, cuff +0.25) but also moves glucose the other way (finger −0.32), and arterial and cuff unequally.

So I'll solve for each state target explicitly, using the head replica's gradients:
- **insulin:** finger = lab = 1, arterial = cuff = 0;
- **NE:** arterial = cuff = 1, finger = lab = 0.

Each solution is restricted to the span of natural directions (nuG, nuM and the top 30 principal components of natural state variation), so the injected change looks like something the model normally represents.

I'll also check two things the b_in amplifier relies on:
- that c_out is well‑conditioned, since the state change is Δy ≈ c_out·Δh_new through the residual coda;
- that the coda Jacobian is close to that linear approximation. I'll test this by numerically inverting the coda replica at real states.

Small residuals on the targets, a moderate state change and a coda close to linear would let me build the full edit next run.

```python
ymean_all=xo[:,192:960].reshape(-1,4,192).mean(1).astype(np.float64)
Gm={}
for ev in EVS:
    d_=HD[ev]; ym=xo[d_['g'].bi.values,192:960].reshape(-1,4,192).mean(1).astype(np.float64)
    base=head_at(d_['rest'],d_['trans'],d_['cen'],1,d_['OUT'],d_['HM'],d_['dt']); L0=LNn(ym,HW['core.out_norm.weight'],HW['core.out_norm.bias']); gg=np.zeros(192)
    for j in range(192):
        e=np.zeros(192); e[j]=1.0
        dO=LNn(ym+e,HW['core.out_norm.weight'],HW['core.out_norm.bias'])-L0
        gg[j]=(head_at(d_['rest'],d_['trans'],d_['cen'],1,d_['OUT']+dO,d_['HM']+e,d_['dt'])-base).mean()
    Gm[ev]=gg
GM=np.stack([Gm[ev] for ev in EVS])   # 4x192
yc=ymean_all-ymean_all.mean(0); _,sv_,Vt_=np.linalg.svd(yc,full_matrices=False); PCs=Vt_[:30].T
Dn=np.column_stack([A(nuG)[0],A(nuM)[0],PCs])
def solve_dir(target):
    GD=GM@Dn; Mi=np.linalg.inv(Dn.T@Dn+1e-6*np.eye(Dn.shape[1])); al=Mi@GD.T@np.linalg.solve(GD@Mi@GD.T,target); return Dn@al
dF=solve_dir(np.array([1.,1.,0.,0.])); dM=solve_dir(np.array([0.,0.,1.,1.]))
for nm,dd in [('dF',dF),('dM',dM),('nuG',A(nuG)[0])]: print(nm,'|d| %.3f'%np.linalg.norm(dd),'effects',np.round(GM@dd,3),' cos nuG %.2f nuM %.2f'%(cs(dd,A(nuG)[0]),cs(dd,A(nuM)[0])))
Wco=Wt('core.c_out.weight'); svc=np.linalg.svd(Wco,compute_uv=False); print('c_out sv max %.3f min %.4f cond %.1f'%(svc[0],svc[-1],svc[0]/svc[-1]))
try:
    Y0=yt_[:200]; hn=np.linalg.solve(Wco,Y0.reshape(-1,192).T).T.reshape(200,4,192)
    cod=lambda h: A(coda_only(torch.tensor(h,dtype=torch.float32))).reshape(h.shape)
    for it in range(30): hn=hn+np.linalg.solve(Wco,(Y0-cod(hn)).reshape(-1,192).T).T.reshape(hn.shape)
    print('coda inversion resid rel %.2e'%(np.linalg.norm(Y0-cod(hn))/np.linalg.norm(Y0)))
    for nm,dd in [('dF',dF),('dM',dM)]:
        v=np.linalg.solve(Wco,dd); dy=(cod(hn+v[None,None,:]*3)-cod(hn))/3
        print(nm,'coda: mean-slot dy vs target cos %.3f ratio %.3f'%(cs(dy.mean(1).mean(0),dd),np.linalg.norm(dy.mean(1).mean(0))/np.linalg.norm(dd)),' effects',np.round(GM@dy.mean(1).mean(0),3))
except Exception as ex: print('coda err',repr(ex)[:300])
```