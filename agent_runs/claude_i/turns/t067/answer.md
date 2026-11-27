Using the empty-set encoding for e_k raises drug-only bundles from 0.50 to 0.73. The other changes fail: interleaved output layouts give ≈0, and unit permutations leave measurement bundles near 0.35. So several pieces are still wrong. Before searching further, I should check my assumptions about the state layout itself, since everything rests on them.

Next, I will run two checks:
1. Whether the stored out equals out_norm(mean of the 4 stored h slots), as the architecture implies. If not, the stored h is not what the next step's time flow acts on.
2. The column norms of z_post's first layer in its three 192-wide blocks, to confirm the input order [ctx, e, u] and which block matters.

An exact match in (1) means the layout is right and the error is in ctx or the encoders. A mismatch would explain most of the trouble at once.

```python
Hall = X3[:,192:960].reshape(-1,4,192)
out_chk = lnorm(Hall.mean(1), P['ow'], P['ob'])
print("out vs out_norm(mean slots(h)): MAD", np.abs(out_chk - X3[:,:192]).mean().round(5), "corr", np.corrcoef(out_chk.ravel(), X3[:,:192].ravel())[0,1].round(6))
W0 = Tn('z_post.0.weight')
print("z_post.0 block col-norm means [ctx,e,u]:", [np.linalg.norm(W0[:, 192*b:192*(b+1)],axis=0).mean().round(3) for b in range(3)])
print("out norm stats: mean|out|", np.abs(X3[:,:192]).mean().round(3), " mean|h|", np.abs(Hall).mean().round(3), " |z|", np.abs(X3[:,960:]).mean().round(3))
```