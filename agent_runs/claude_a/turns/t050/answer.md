Two new facts. First, the stored out equals the normalised mean of the slots, so the undecayed ctx is just 2·out + gap. Second, even a ridge regression on ctx and e predicts the stored z only to an error of 0.026. Nothing I have tried reproduces z_post.

Before I give up on the core rebuild, one cheap check remains: the stored z might be shifted by one bundle relative to the state it is stored with. I will compare my z_post output with the stored z of the previous and next bundle. If one of these matches, the misalignment explains all my failures.

In the same run I will compute the true effects for the Forward-1 tuning stays. For each counted stay-hour, the true shift is (multiplier − 1) times the sum over lags of my causal kernel times the recorded rates from hour 12 on. This tells me how far the model's −1.88 is from the target.

```python
o_nd = mlp(torch.cat([ctx1['nd'], e1, Z1], 1), 'z_post', F.gelu)[:, :32]
o_d = mlp(torch.cat([ctx1['d'], e1, Z1], 1), 'z_post', F.gelu)[:, :32]
ix = torch.tensor(n1_rows)
for sh in [-1, 0, 1]:
    print('shift', sh, 'nd', round((o_nd - ZP[ix + sh]).abs().mean().item(), 4), 'd', round((o_d - ZP[ix + sh]).abs().mean().item(), 4))
# true effects for Forward-1 stays
import os
d1 = pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print(d1.columns.tolist(), d1.shape); print(d1.head(3).to_string())
print('psi_ins', np.round(psi_ins, 2), 'psi_ne', np.round(psi_ne, 2))
```