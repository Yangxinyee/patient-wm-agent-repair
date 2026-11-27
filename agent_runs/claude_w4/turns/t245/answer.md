Two useful results:
1. My prelude reimplementation reproduces the stored prelude outputs exactly (error 4e-6). Head-1 attention from slots to the action token averages 0.998 but drops to 0.28 at some bundles.
2. The Jacobian of `out` with respect to an injection is essentially identical whether or not the age embedding is added (median cos 0.9998). This holds even though both variants fit the levels poorly. So the memory approximation doesn't matter for designing an injection direction.

The Jacobian's size is the important news. Injecting along the head-1 output directions moves `out` by only about 0.0003 per unit of norm. This explains why every prelude-level edit so far had almost no effect. The likely reason is that `out` is built mainly from cross-attention over the memory of past event tokens, which hold the readings but not the doses. The dose enters only through the action token in the prelude, a path the trained model has learned to barely use.

Whether a usable edit exists depends on the best possible direction, not the average one. Next I'll compute, on 120 drug bundles, the full Jacobian of `out` with respect to any 192-dim vector added to the slots after prelude attention. Any such vector is reachable by remapping head-1 output columns along a value direction the original computation leaves unused.

I'll combine it with the ridge readouts from `out` for finger glucose and arterial MAP. That gives:
- the gradient directions that move each prediction most per unit of injection;
- the injection norm needed for a 1 mg/dL or 1 mmHg shift;
- how separable glucose and MAP effects are.

I'll also report where the shrinkage happens: the prelude norm, the new slots, or `out`.
- **Viable:** a few units of norm per mg/dL, versus slot norms near 800.
- **Not viable without large nonlinear perturbations:** hundreds.

```python
with torch.no_grad():
    sel2=sel[:120]; Tb=TOKt[sel2]; E_b=prelude_tail(Tb); hb,ob=step_full(di[sel2],E_b.float(),htft[sel2],pre=True)
    Jx=np.zeros((len(sel2),192,192)); JxE=np.zeros(len(sel2)); JxH=np.zeros(len(sel2))
    h=2.0
    for j in range(192):
        dx=torch.zeros(len(sel2),4,192,dtype=torch.float64); dx[:,:,j]=h
        Ep=prelude_tail(Tb,dx); Em=prelude_tail(Tb,-dx)
        hp_,op=step_full(di[sel2],Ep.float(),htft[sel2],pre=True); hm_,om=step_full(di[sel2],Em.float(),htft[sel2],pre=True)
        Jx[:,:,j]=np.asarray(op-om)/(2*h)
        JxE+=np.asarray(((Ep-Em)/(2*h)).norm(dim=(1,2)))**2; JxH+=np.asarray(((hp_-hm_)/(2*h)).norm(dim=(1,2)))**2
print('per unit dx (same in 4 slots, norm 2 per basis vec): rms |dE| %.4f  rms |dhn| %.3f  rms |dout| %.5f'%(np.sqrt(JxE.mean()/192),np.sqrt(JxH.mean()/192),np.sqrt((Jx**2).sum((1,2)).mean()/192)))
wF=WR_fit['glucose_finger'].coef_; wA=WR_fit['map_arterial'].coef_; wL=WR_fit['glucose_lab'].coef_; wC=WR_fit['map_cuff'].coef_
print('|wF| %.2f |wA| %.2f (mg/dL or mmHg per unit out)'%(np.linalg.norm(wF),np.linalg.norm(wA)))
gF=np.einsum('o,noj->nj',wF,Jx); gA=np.einsum('o,noj->nj',wA,Jx)
gFm=gF.mean(0); gAm=gA.mean(0)
print('mean-gradient |gF| %.4f mg/dL per unit dx-norm (per-bundle |gF| median %.4f); |gA| %.4f mmHg (median %.4f)'%(np.linalg.norm(gFm),np.median(np.linalg.norm(gF,axis=1)),np.linalg.norm(gAm),np.median(np.linalg.norm(gA,axis=1))))
print('consistency across bundles: cos(gF_n, gFm) median %.3f ; cos(gA_n,gAm) median %.3f ; cos(gFm,gAm) %.3f'%(np.median(gF@gFm/np.linalg.norm(gF,axis=1)/np.linalg.norm(gFm)),np.median(gA@gAm/np.linalg.norm(gA,axis=1)/np.linalg.norm(gAm)),gFm@gAm/np.linalg.norm(gFm)/np.linalg.norm(gAm)))
print('=> dx norm needed for 1 mg/dL: %.1f ; for 1 mmHg: %.1f'%(1/np.linalg.norm(gFm),1/np.linalg.norm(gAm)))
sv=np.linalg.svd(Jx.mean(0),compute_uv=False); print('mean Jx singular values top5',np.round(sv[:5],4),' median',np.round(np.median(sv),5))
ro=np.linalg.matrix_rank(W1o_t.numpy()); P_W=W1o_t.numpy()@np.linalg.pinv(W1o_t.numpy()); print('fraction of |gFm| in range(W1o): %.3f, of |gAm|: %.3f'%(np.linalg.norm(P_W@gFm)/np.linalg.norm(gFm),np.linalg.norm(P_W@gAm)/np.linalg.norm(gAm)))
```