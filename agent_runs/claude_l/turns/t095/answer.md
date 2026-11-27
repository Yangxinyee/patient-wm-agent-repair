Even with a free context offset, the fit cannot reproduce z: the RMS residual is 0.07–0.11 and the maximum about 1. So either my measurement encodings are wrong, or e does not enter z_post where I assumed. The simplest structural alternative is a different input order. For example, the order might be [ctx, u, e], or [e, ctx, u], instead of [ctx, e, u]. With u = 0 at these bundles, only the position of e matters.

Next I refit the free context offset with e placed in each of the three 192-column blocks of `z_post.0.weight`, for a few activation choices. If one placement gives a near-zero residual, the input order is found, and both the encoder and the z_post activation are confirmed. If all placements still fail, the encoder output itself is wrong.

```python
for blk in [0,1,2]:
    for ea,za in [('silu','silu'),('gelu','gelu'),('gelu','silu'),('silu','gelu')]:
        ACTMAP.update({'encode.mlp1':ea,'encode.mlp2':ea,'encode_action.mlp1':ea,'encode_action.mlp2':ea})
        Ev=np.stack([enc_bundle(BUN[sid_arr[fi[i]]][1][1])[0].numpy() for i in idx1])
        V=Ev@W1[:,192*blk:192*(blk+1)].T + b1
        def resid(a):
            return ((act_np(za,a[None,:]+V)@W2.T + b2) - Zi).ravel()
        r=least_squares(resid, np.zeros(192), method='lm', max_nfev=3000)
        print(blk,ea,za,'rms %.5f max %.5f'%(np.sqrt(np.mean(r.fun**2)), np.abs(r.fun).max()))
```