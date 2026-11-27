Both anchors hold:
- `out` is exactly `out_norm(mean over slots of h)`, with a maximum error of 1e-6.
- The first bundle of every stay gives the same state and the same z across stays, to within 1e-6. It is a single hospital-admission atom at hour −2.

That first bundle is a fully deterministic test case. The only unknowns are constants: the atom embedding, the two encoder MLPs, ctx at the start (initial state and initial dt), and the activations. The z spread per dimension ranges from 0.04 to 0.59, so no z dimension is completely unused, though dimensions 8–10, 12 and 28 are nearly so.

To narrow the embedding search, I'll inspect the row norms of every embedding table. Zero rows show which index means "absent", which matters for the categorical tables `kq` and `text`. I'll also inspect the structure of the gap, size and age parameters. If the absent rows are exactly zero, those terms drop out for the admission atom and the search shrinks to the activations and the start-of-stay conventions.

```python
tabs = ['embed.code.weight','embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight',
        'embed.bins.value.weight','embed.bins.dose.weight','embed.bins.rate.weight',
        'embed.categorical.action.weight','embed.categorical.discharged_to.weight','embed.categorical.rate_unit.weight',
        'embed.categorical.route.weight','embed.categorical.status.weight','embed.categorical.unit.weight',
        'type_emb.weight','last_status.weight','core.slots']
for t_ in tabs:
    W_ = load(t_); print(t_, tuple(W_.shape), 'row norms', np.round(W_.norm(dim=1).numpy(), 3))
nw = load('embed.numeric.weight'); print('numeric col norms', np.round(nw.norm(dim=0).numpy(), 3), 'bias norm', load('embed.numeric.bias').norm().item())
for t_ in ['encode.size.weight','encode.size.bias','encode_action.size.weight','encode_action.size.bias','gap.0.weight','gap.0.bias','core.flow.a','core.a_in','core.delta_in','z_in.bias','z_head.bias']:
    W_ = load(t_); print(t_, tuple(W_.shape), 'norm %.3f' % W_.norm().item(), np.round(W_.ravel()[:5].numpy(), 3))
fa_ = load('core.flow.a').numpy(); print('flow timescales (h) quantiles', np.round(np.quantile(1/np.exp(fa_)/3600, [0, .1, .25, .5, .75, .9, 1]), 3))
```