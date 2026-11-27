No combination matches exactly even at k=1. So my memory of an exact match in run 66 was wrong: it measured something else. The ctx/z formula as I understand it from the documentation is off somewhere.

I will rebuild from solid anchors:
1. Check that the stored `out` equals `out_norm(mean over slots of h)` in every row. This validates the layout and the norm.
2. Look at `h0`/`out0`, which may be an initial state I found earlier.
3. Run the documented step on the first bundle from a zero state, compare all of `out`, `h` and `z`, and print the size of each intermediate.

If identity 1 holds, the state layout is right and the error lies inside the step. The first-bundle test, with an empty memory and a fixed content, isolates the time flow, gap, z_post, prelude and coda without the cross-attention unknowns.

```python
Xt=torch.tensor(X,dtype=T)
outX=Xt[:,:192]; hX=Xt[:,192:960].reshape(-1,4,192)
chk=lnorm(W0,hX.mean(1),'core.out_norm')
print('out==out_norm(mean h)?', (chk-outX).abs().max().item())
print('out0 vs X0', (out0.double()-outX[0]).abs().max().item(), '| h0 vs X0 h', (h0.double()-hX[0]).abs().max().item(), '| h0 vs slots', (h0.double()-W0['core.slots']).abs().max().item())
print('all stays same row0?', (Xt[[row_of[s][0] for s in stays]]-Xt[0]).abs().max().item())
print('h0 norm',h0.norm().item(),'slots norm',W0['core.slots'].norm().item(), 'hX0 norm', hX[0].norm().item())
print('flow tau range (s):', (1/torch.exp(W0['core.flow.a'])).min().item(), (1/torch.exp(W0['core.flow.a'])).max().item())
sp=F.softplus(W0['core.delta_in']); abar=torch.exp(-sp*torch.exp(W0['core.a_in']))
print('abar range', abar.min().item(), abar.max().item(), 'sp range', sp.min().item(), sp.max().item())
```