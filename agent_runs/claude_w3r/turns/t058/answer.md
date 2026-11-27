Neither tie rule fixes bundles 9 and 11; the error stays at about 0.17. The insulin rate edges contain duplicates (0.5 three times, 1.0 four times), so 0.5 u/h maps to bin 1 under one rule and bin 4 under the other, and both are wrong. Something else about the 0.5 u/h atom differs from my encoding.

Next I brute-force the rate bin and the dose bin of that insulin start atom in bundle 9, trying every pair of indices 0 to 16 and keeping the lowest error. If one pair brings the error to about 0.03, the model's binning rule for duplicated edges differs from both of mine, and that pair tells me the rule. If no pair helps, the problem is in the z-score or numeric term.

```python
def zk_custom(k, atom_mod, o=UB):
    out_=torch.tensor(X0[k-1,:192]); h_=torch.tensor(X0[k-1,192:960]).reshape(4,192)
    dt=(bbB[k][0]-bbB[k-1][0])*3600; hf=h_*torch.exp(-dt*torch.exp(fa))
    ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
    at=bbB[k][1]; ac=[a for a in at if a['ek']==2]
    vs=[atom_mod(a, emb(a,o)) for a in ac]
    u_=enc(torch.stack(vs),'encode_action','gelu'); e_=torch.zeros(192)
    return MLP(torch.cat([ctx,e_,u_]),'z_post','gelu')[:32]
best=[]
zt9=torch.tensor(X0[9,960:])
for br in range(17):
    for bd in range(17):
        def mod(a,v,br=br,bd=bd):
            if a['code']==8 and a['kind']==0:
                v=v-Wo['embed.bins.rate.weight'][fbin('insulin','rate',0.5)]-Wo['embed.bins.dose.weight'][fbin('insulin','dose',0.5)]
                v=v+Wo['embed.bins.rate.weight'][br]+Wo['embed.bins.dose.weight'][bd]
            return v
        best.append((round(float((zk_custom(9,mod)-zt9).abs().mean()),4),br,bd))
best.sort(); print(best[:6])
```