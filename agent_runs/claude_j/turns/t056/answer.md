The z posterior is not matching at all. The model's z values have a typical size of only about 0.14–0.23. My error of 0.28 per bundle is as large as the signal, and the encoder and context variants barely change it. So I am probably not reading the right output or using the right combination of networks.

To find the mapping, I'll correlate each of the model's 32 z dimensions with all 64 outputs of my z_post computation and all 64 of z_prior. I'll also try combinations such as prior mean plus a posterior correction, and a precision-weighted blend. A clear one-to-one correlation near 1 would reveal the true formula. Low correlations everywhere would mean the inputs themselves (context or encoder) are wrong.

```python
rows_ = []
for r_, rp, dt, ms_ in tests[:200]:
    e_ = encode([meas_atom(c, v) for c, v in ms_])
    out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
    dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
    ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([math.log(dt)/10]), 'gap')
    zpo = mlp(torch.cat([ctx, e_, torch.zeros(192)]), 'z_post')
    zpr = mlp(ctx, 'z_prior')
    rows_.append((Xs[r_,960:992].numpy(), zpo.numpy(), zpr.numpy()))
Zs = np.stack([a for a,_,_ in rows_]); ZPO = np.stack([b for _,b,_ in rows_]); ZPR = np.stack([c for _,_,c in rows_])
def bestcorr(A, B):
    C = np.corrcoef(np.concatenate([A,B],1).T)[:A.shape[1], A.shape[1]:]
    return C
C1 = bestcorr(Zs, ZPO); C2 = bestcorr(Zs, ZPR)
print('post: diag corr (z_d vs out_d)', np.round(np.diag(C1[:, :32]),2))
print('post: argmax', np.abs(C1).argmax(1))
print('prior: diag corr', np.round(np.diag(C2[:, :32]),2))
# try combos: prior mean + post mean ; precision-weighted
pm, pls = ZPR[:,:32], ZPR[:,32:]; qm, qls = ZPO[:,:32], ZPO[:,32:]
for nm_, cand in [('post',qm), ('prior',pm), ('prior+post', pm+qm), ('prec', (pm*np.exp(-2*pls)+qm*np.exp(-2*qls))/(np.exp(-2*pls)+np.exp(-2*qls)))]:
    print(nm_, 'mean abs err', np.abs(cand-Zs).mean().round(4), 'corr', np.corrcoef(cand.ravel(), Zs.ravel())[0,1].round(3))
```