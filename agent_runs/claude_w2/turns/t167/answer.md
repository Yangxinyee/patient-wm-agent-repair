Even the full core state (out plus all 4 slots, 960 numbers) carries the dose only partially. Lag-1 insulin decodes with held-out R² 0.78 and counterfactual slope 0.69, lag 2 with 0.60 / 0.50, lag 3 with 0.55 / 0.40. So the model keeps only a blurred memory of the dose, and no readout of the existing state can produce the structural response. An accurate fix needs a new, clean dose channel inside the core.

Doses enter the core only through the action vector u, which becomes a token of the prelude transformer layer (directly, and via z). So any dose channel must pass from the u token into the 4 slot tokens through prelude attention. Whether that is workable depends on the attention weights. If some head's slot queries attend to the u token strongly and consistently, a dose signal can be carried through that head's value vector with little multiplicative noise. I'll measure the prelude attention weights from the slot queries to each token at insulin bundles (exact states, selT).

```python
def prelude_attn(idx):
    hp=h_t[idx-1]; op=out_t[idx-1]
    dt=torch.as_tensor((B_T.t.values[idx]-B_T.t.values[idx-1])*3600.0,dtype=torch.float32)
    hf=hp*torch.exp(-dt[:,None,None]*torch.exp(W['core.flow.a']))
    X=torch.cat([hf+W['core.slots'][None],E_T[idx][:,None],U_T[idx][:,None],F.linear(z_t[idx],W['z_in.weight'],W['z_in.bias'])[:,None]],1)
    xn=LN(X,W,'core.prelude.0.norm1')
    q=F.linear(xn,W['core.prelude.0.attn.q.weight']); kv=F.linear(xn,W['core.prelude.0.attn.kv.weight']); k=kv[...,:192]
    B_,L_,_=q.shape
    q=q.reshape(B_,L_,4,48).transpose(1,2); k=k.reshape(B_,L_,4,48).transpose(1,2)
    a=((q@k.transpose(-1,-2))/math.sqrt(48)).softmax(-1)   # B,4heads,7q,7k
    return a
sidB=B_T.stay_id.values; okprev=np.r_[False,sidB[1:]==sidB[:-1]]
# insulin bundles: those with an insulin START atom
ins_b=set(zip(A_T[(A_T.code==CODE['insulin'])&(A_T.kind==0)].stay_id, A_T[(A_T.code==CODE['insulin'])&(A_T.kind==0)].bidx))
isI=np.array([(s,b) in ins_b for s,b in zip(B_T.stay_id.values,B_T.bidx.values)])&okprev
noA=(~hasu_T)&okprev
for nm_,msk in [('insulin bundles',isI),('no-action bundles',noA)]:
    a=prelude_attn(np.where(msk)[0])
    print(nm_, msk.sum())
    for hd in range(4):
        m_=a[:,hd,:4,:].mean(0).mean(0).numpy()   # avg over slot queries
        s_=a[:,hd,:4,5].std(0).mean().item()
        print('  head',hd,'attn on tokens [s0..s3,e,u,z]:',m_.round(3),' sd(u)',round(s_,3))
```