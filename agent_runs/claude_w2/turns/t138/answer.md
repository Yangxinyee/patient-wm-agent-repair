The coordinate search over column layouts only overfits: it repeats features in odd places and still leaves an error of 2.0. So the 8-column layout is not the main problem.

A cleaner clue is the empty-register first readings. MAP is accurate there but glucose is not, from the same state, so some code-dependent term in with_code is probably missing or different. Next I try adding atom-style terms to with_code:
- the code's unit row;
- its rate-unit row;
- the kind and event-kind rows;
- kq[0];
- the numeric bias;
- their combination (a full "attribute-less atom" embedding of the code).

I score each by error and per-code bias. A term that removes the glucose bias at empty-register rows without hurting MAP is the missing piece.

```python
URt=W['embed.categorical.unit.weight']; RRt=W['embed.categorical.rate_unit.weight']
cvals=fa.code.values
cand_vecs={
 'unit':URt[torch.as_tensor(UROW[cvals])],
 'rateunit':RRt[torch.as_tensor(RROW[cvals])],
 'kind0':W['embed.kind.weight'][0].expand(len(fa),-1),
 'ekind1':W['embed.ekind.weight'][1].expand(len(fa),-1),
 'kq0':W['embed.kq.weight'][0].expand(len(fa),-1),
 'numbias':W['embed.numeric.bias'].expand(len(fa),-1),
}
cand_vecs['atom_noattr']=sum(cand_vecs[k] for k in ['unit','rateunit','kind0','ekind1','kq0','numbias'])
f_=np.stack([FE[c] for c in ['has','la_h','zl','has','zero','zero','zero','zero']],1)
lnum=F.linear(torch.as_tensor(f_),W['last_numeric.weight'],W['last_numeric.bias'])
for nm_,vec in [('none',0)]+list(cand_vecs.items()):
    wc=base_wc+lnum+vec
    pm=(torch.softmax(mlp(wc,W,'attr_heads.value_bin')+trans,-1)[:,1:].numpy()*cm).sum(1); d=pm-fa.predicted_mean.values
    print(nm_, round(np.abs(d).mean(),3), fa.assign(d=d).groupby(['event','has']).d.mean().round(2).values)
```