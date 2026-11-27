dt is about one hour for every reading, as expected; the bundles before them are action or reading bundles 55 to 60 minutes earlier. The errors are not uniform. Readings with a filled last register show a systematic bias, especially MAP (+2.6 to +3.0), while glucose is mixed (+1.0 finger, −2.3 lab). Readings with an empty register are off less.

The tool averages probabilities over only 4 prior draws of z. That adds zero-mean noise to its predicted means but cannot create a bias. So two checks:
1. Estimate how large that noise is, giving the error floor I can expect.
2. Test whether the last register is shared within each code pair, using the stored fields for the most recent reading of either code in the pair.

If the noise explains the mean absolute error and the shared register removes the bias, the head is solved.

```python
W=WO
hf=Hs_[Pp]*torch.exp(-dtt[:,None,None]*torch.exp(W['core.flow.a']))
base=OUTs[Pp]+LN(hf.mean(1),W,'core.out_norm'); ctx=base+mlp(torch.log(dtt)[:,None]/10,W,'gap')
zo=mlp(ctx,W,'z_prior'); mu,lsd=zo[:,:32],zo[:,32:]
print('prior sd mean', float(torch.exp(lsd).mean()))
def pm_from(z,f8,lb,codes):
    q=LN(ctx+F.linear(z,W['z_head.weight'],W['z_head.bias'])+W['type_emb.weight'][1],W,'query_norm')
    wc=q+W['embed.code.weight'][codes]+F.linear(torch.as_tensor(f8),W['last_numeric.weight'],W['last_numeric.bias'])+W['embed.bins.value.weight'][lb]
    lg=mlp(wc,W,'attr_heads.value_bin')+W['embed_transition.value_bin.weight'][codes*17+lb]
    return torch.softmax(lg,-1)
g=torch.Generator().manual_seed(1)
reps=[]
for r in range(20):
    P=sum(pm_from(mu+torch.exp(lsd)*torch.randn(mu.shape,generator=g),f8m,lb_,codes_) for _ in range(4))/4
    reps.append((P[:,1:].numpy()*cm).sum(1))
reps=np.array(reps); Epm=reps.mean(0)
print('4-draw noise sd', round(float(reps.std(0).mean()),3), 'expected mae rep-vs-rep', round(float(np.abs(reps[0]-reps[1]).mean()),3))
d=Epm-fa.predicted_mean.values
print('E-sample vs tool mae', round(float(np.abs(d).mean()),3), 'bias', round(float(d.mean()),3))
print('by event', fa.assign(d=d).groupby('event').d.agg(['mean',lambda x: np.abs(x).mean()]).round(3).to_dict())
```