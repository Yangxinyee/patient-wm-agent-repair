Type 1 is best for every code, so the type index is not the problem. I still have mean errors of 2.6 to 3.2 with code-specific biases.

Next I check, rather than assume, the remaining terms ARCHITECTURE.md lists for with_code:
- the norms of last_status rows 0/1 and bins.rate row 0 (if any is non-zero, it belongs in the sum);
- whether adding a "no previous value" column of last_numeric fixes the empty-register glucose rows;
- how the error for filled-register rows depends on the time since the previous bundle and on whether a drug bundle preceded the reading.

A non-zero status row that removes the biases would settle it. A pattern tied to preceding drug bundles would instead point to the state or dt that the head reads.

```python
print('last_status row norms', W['last_status.weight'].norm(dim=1).numpy().round(3), 'bins.rate[0] norm', float(W['embed.bins.rate.weight'][0].norm()), 'bins.value[0]', float(W['embed.bins.value.weight'][0].norm()))
print('last_numeric col norms', W['last_numeric.weight'].norm(dim=0).numpy().round(3))
def eval_wc(extra_vec=None, mask=None):
    q=LN(ctx+F.linear(mu,W['z_head.weight'],W['z_head.bias'])+W['type_emb.weight'][1],W,'query_norm')
    wc=q+W['embed.code.weight'][codes_]+F.linear(torch.as_tensor(f8m),W['last_numeric.weight'],W['last_numeric.bias'])+W['embed.bins.value.weight'][lb_]
    if extra_vec is not None: wc=wc+extra_vec*(torch.as_tensor(mask,dtype=torch.float32)[:,None] if mask is not None else 1)
    lg=mlp(wc,W,'attr_heads.value_bin')+W['embed_transition.value_bin.weight'][codes_*17+lb_]
    return (torch.softmax(lg,-1)[:,1:].numpy()*cm).sum(1)-fa.predicted_mean.values
for nm_,vec,mk in [('st1_all',W['last_status.weight'][1],None),('st0_all',W['last_status.weight'][0],None),
                   ('st1_has',W['last_status.weight'][1],fa.has.values),('st0_has',W['last_status.weight'][0],fa.has.values)]+[('nohas_col%d'%k,W['last_numeric.weight'][:,k],1-fa.has.values) for k in range(8)]:
    d=eval_wc(vec,mk); 
    print(nm_, round(np.abs(d).mean(),3), fa.assign(d=d).groupby(['event','has']).d.mean().round(2).values)
```