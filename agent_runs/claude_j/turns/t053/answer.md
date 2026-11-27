That run died on a dtype clash: the states array loaded as float32 while the weights are float64. Nothing was tested. I'll cast the states to float64 and rerun the same two checks. These are the out = out_norm(mean over slots of h) identity, and the grid of activation function × gap-input scaling for the z posterior. I expect the identity to hold to about 1e-5. If one variant brings the z error near zero, it identifies the activation and the gap convention.

```python
Xs = torch.tensor(S['x'], dtype=torch.float64)
o_chk = (Xs[:, :192] - ln(Xs[:, 192:960].reshape(-1,4,192).mean(1), 'core.out_norm')).abs().max().item()
print('out == out_norm(mean_slots(h)) max err', o_chk)
for an, af in acts.items():
    for gm in ['s10','s','h10','m10','h']:
        print(an, gm, z_err2(30, af, gm))
```