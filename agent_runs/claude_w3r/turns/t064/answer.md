My implementation reproduces the pattern only partly. At 1.5 u/h it matches (1.07 against 1.09). At 1.0 and 2.0 it is weaker. At 0.5 and 3.0 it is far off: 0.32 against −1.12, and 0.04 against −1.09. So the model separates the rates more sharply than my embedding does.

My earlier brute force over bins used the wrong setup: END atoms sat among the actions and e_k was zeroed. I will now redo it with END atoms in the observation encoder, the better variant, over all rate-bin and dose-bin pairs for the 0.5 u/h atom in bundle 9. If some pair reproduces −1.1/−1.3, the model's bin rule for duplicated edges differs from mine, and the winning pair tells me the rule.

```python
def zk_end(k, mod=None, o=UB, Xs_=X0, bb=bbB):
    out_=torch.tensor(Xs_[k-1,:192]); h_=torch.tensor(Xs_[k-1,192:960]).reshape(4,192)
    dt=(bb[k][0]-bb[k-1][0])*3600; hf=h_*torch.exp(-dt*torch.exp(fa))
    ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
    at=bb[k][1]; ob=[a for a in at if a['ek']!=2 or a['kind']==2]; ac=[a for a in at if a['ek']==2 and a['kind']!=2]
    ev_=lambda a: mod(a,emb(a,o)) if mod else emb(a,o)
    e_=enc(torch.stack([ev_(a) for a in ob]),'encode','gelu') if ob else torch.zeros(192)
    u_=enc(torch.stack([ev_(a) for a in ac]),'encode_action','gelu') if ac else torch.zeros(192)
    return MLP(torch.cat([ctx,e_,u_]),'z_post','gelu')[:32]
best=[]
for br in range(17):
    for bd in range(17):
        def mod(a,v,br=br,bd=bd):
            if a['code']==8 and a['kind']==0:
                v=v-Wo['embed.bins.rate.weight'][fbin('insulin','rate',0.5)]-Wo['embed.bins.dose.weight'][fbin('insulin','dose',0.5)]+Wo['embed.bins.rate.weight'][br]+Wo['embed.bins.dose.weight'][bd]
            return v
        zp=zk_end(9,mod); best.append((round(float((zp-zt9).abs().mean()),4),br,bd,round(float(zp[9]),2),round(float(zp[20]),2)))
best.sort(); print(best[:5]); print('default', round(float((zk_end(9)-zt9).abs().mean()),4))
```