The state does carry dose history, but only partly in linear form. The current insulin rate reads out with cross-validated R² 0.82, older lags fall to 0.61, and the kernel-weighted cumulative effect reaches 0.90. Norepinephrine reaches only about 0.65. Part of this is probably indirect, through the glucose level that also sets the dose. So a head-side correction from a linear readout would be imprecise, especially for doubled doses outside the training range.

I'll take the more reliable route: rebuild the model's step offline, check it against the fwd_3 states, then design and test edits with offline rollouts. Validation goes bottom-up. The first and cheapest check is whether the reported out equals core.out_norm applied to the mean of the 4 reported h slots. An exact match (error near 1e-5) would confirm the state layout and LayerNorm convention. A mismatch would mean out comes from a different quantity.

```python
import torch
torch.set_grad_enabled(False)
Wt = lambda n: torch.tensor(load(n).numpy(), dtype=torch.float64)
x3 = torch.tensor(X3s)
out3 = x3[:, :192]; h3 = x3[:, 192:960].reshape(-1, 4, 192); zm3 = x3[:, 960:]
LN = torch.nn.functional.layer_norm
o_chk = LN(h3.mean(1), (192,), Wt('core.out_norm.weight'), Wt('core.out_norm.bias'), eps=1e-5)
print('out vs LN(mean h):', (o_chk - out3).abs().max().item(), (o_chk - out3).abs().mean().item())
print('h norms', h3.norm(dim=-1).mean().item(), 'out norm', out3.norm(dim=-1).mean().item(), 'z sd', zm3.std().item())
```