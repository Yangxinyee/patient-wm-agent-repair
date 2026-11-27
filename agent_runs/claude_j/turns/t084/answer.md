The relation `out = out_norm(mean_slots(h))` holds exactly, so the state layout is right. The h entries are large, though: mean absolute value 43, up to 358. An error of 20–35 therefore means my step is substantially wrong, not slightly off. Before testing structural variants, I'll look at the scale-setting parameters: the time-flow rates `exp(core.flow.a)` (the documented timescales of 10 minutes to 2 years would put these between about 1.6e-8 and 1.7e-3 per second), the gate parameters, and how close `c_out` is to the identity. A mismatch here would show that a parameterisation differs from the documented one.

```python
fa = W['core.flow.a']; print('exp(flow.a) range', torch.exp(fa).min().item(), torch.exp(fa).max().item(), ' 1/exp range', (1/torch.exp(fa)).min().item(), (1/torch.exp(fa)).max().item())
sp = Fn.softplus(W['core.delta_in']); print('softplus(delta_in)', sp.min().item(), sp.max().item())
abar = torch.exp(-sp*torch.exp(W['core.a_in'])); print('a_bar', abar.min().item(), abar.median().item(), abar.max().item())
C_ = W['core.c_out.weight']; print('c_out diag mean', torch.diag(C_).mean().item(), '|C-I|/|I|', (torch.linalg.norm(C_-torch.eye(192))/math.sqrt(192)).item(), 'svals', torch.linalg.svdvals(C_)[[0,10,50,100,191]].numpy())
print('slots norm', W['core.slots'].norm(dim=1).numpy(), 'b_in norm', torch.linalg.matrix_norm(W['core.b_in.weight']).item())
# k=0: state after first bundle, try h0=0
print('first bundle hour', hr_[idx[0]], 'state h abs mean at k=0', Xt[idx[0],192:960].abs().mean().item(), 'k=1', Xt[idx[1],192:960].abs().mean().item())
```