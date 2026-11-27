The baseline error alternates with bundle type. It is 0.39–0.57 at reading bundles (integer hours, dt ≈ 3300 s) and 0.05–0.15 at drug-start bundles (dt = 300 s). My notes say the inputs were checked against a linear probe. So the mismatch is probably in how the posterior mean is formed from the network outputs, not in ctx, e or u. Possibilities are a residual on the prior mean, a precision-weighted combination of prior and posterior, or the other half of the 64 outputs. I'll compute z_post and z_prior for a few stays and test each.

```python
fa0 = W0['core.flow.a']
def zparts(s, gapf=lambda dt: math.log(dt)/10):
    i0 = row_start[s]; B_ = BUN[s]; R=[]
    for k in range(1, len(B_)):
        prev = torch.tensor(Xs[i0+k-1], dtype=T); out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
        dt = (B_[k][0]-B_[k-1][0])*3600.0
        e,u = enc_eu2(B_[k][1])
        hf = h_p*torch.exp(-dt*torch.exp(fa0)); g = mlp(W0, torch.tensor([gapf(dt)],dtype=T), 'gap')
        ctx = out_p + lnorm(W0, hf.mean(0), 'core.out_norm') + g
        zq = mlp(W0, torch.cat([ctx,e,u]), 'z_post'); zpr = mlp(W0, ctx, 'z_prior')
        R.append((zq.numpy(), zpr.numpy(), Xs[i0+k,960:992], dt))
    return R
R=[]; 
for s in ks[:6]: R += zparts(s)
ZQ=np.array([r[0] for r in R]); ZP=np.array([r[1] for r in R]); ZR=np.array([r[2] for r in R]); DT=np.array([r[3] for r in R])
cands = {'q_mean':ZQ[:,:32], 'q_2nd':ZQ[:,32:], 'q+prior':ZQ[:,:32]+ZP[:,:32], 'prior':ZP[:,:32]}
vq=np.exp(2*ZQ[:,32:]); vp=np.exp(2*ZP[:,32:]); cands['prec_w']=(ZQ[:,:32]/vq+ZP[:,:32]/vp)/(1/vq+1/vp)
vq1=np.exp(ZQ[:,32:]); vp1=np.exp(ZP[:,32:]); cands['prec_w_logvar']=(ZQ[:,:32]/vq1+ZP[:,:32]/vp1)/(1/vq1+1/vp1)
for k_,v_ in cands.items():
    err=np.abs(v_-ZR); print(f"{k_:14s} median {np.median(err):.4f} max {err.max():.3f}  med@reading {np.median(err[DT>1000]):.4f} med@drug {np.median(err[DT<1000]):.4f}")
```