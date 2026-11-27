Even restricted to the 21 single arterial-MAP bundles, where only the reading's value varies, the fit leaves an RMS residual of about 0.0065 rather than about 1e-6. Either the value encoding (z-score or bin) differs subtly, or the 192-parameter fit simply has not converged in 200–300 evaluations.

Next I rule out the convergence question first, because it is cheap. I rerun the subset fit with many more iterations, using the Levenberg–Marquardt method with the analytic Jacobian, and also from a second starting point. If the residual drops to about 1e-6, the encoder is right for this atom type and the earlier "mismatches" were partly an optimisation artefact. If it stays near 0.006, the value encoding really differs.

```python
ea,za='gelu','silu'
ACTMAP.clear(); ACTMAP.update({'encode.mlp1':ea,'encode.mlp2':ea})
Ev7=np.stack([set_enc(W0,[a for a in BUN[k1[i]][1][1] if not a['act']],'encode').numpy() for i in single7])
Vv7=Ev7@W1[:,192:384].T+b1; Z7=Zi[single7]
fun=lambda a: ((act_np(za,a[None,:]+Vv7)@W2.T+b2)-Z7).ravel()
jac=lambda a: np.concatenate([W2*dact_np(za,a+Vv7[i])[None,:] for i in range(len(single7))],0)
for init in ['zero','rand']:
    a0_=np.zeros(192) if init=='zero' else np.random.RandomState(0).randn(192)
    t0=time.time(); r=least_squares(fun,a0_,jac=jac,method='lm',max_nfev=20000,xtol=1e-14,ftol=1e-14,gtol=1e-14)
    print(init,'rms %.3e'%np.sqrt(np.mean(r.fun**2)),'nfev',r.nfev,'status',r.status,'%.0fs'%(time.time()-t0))
print('values:', sorted(set((a['vb'],round(a['num'][0],3)) for i in single7 for a in BUN[k1[i]][1][1])))
```