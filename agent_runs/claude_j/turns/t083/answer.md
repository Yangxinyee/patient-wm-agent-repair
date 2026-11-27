Every variant is off by 20–35, so something basic is wrong. The state layout or the update formulas don't match my reading. I'll check the simplest relation first, using the GPU states alone: whether `out` equals `out_norm(mean over slots of h)` for the same bundle. I'll also look at the typical size of the h entries. If that relation holds, the error is inside the step and I'll test it piece by piece. If not, the stored h is something other than the coda output.

```python
o_g = Xt[:, :192]; h_g = Xt[:, 192:960].view(-1,4,192)
print('out abs mean', o_g.abs().mean().item(), 'h abs mean', h_g.abs().mean().item(), 'h max', h_g.abs().max().item())
o_hat = lnp(h_g.mean(1), 'core.out_norm')
print('out vs out_norm(mean h):', (o_hat-o_g).abs().max().item())
# maybe out_norm of mean of something else; check correlation per row
print('corr', np.corrcoef(o_hat[:200].reshape(-1).numpy(), o_g[:200].reshape(-1).numpy())[0,1])
```