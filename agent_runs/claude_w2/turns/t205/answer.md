The head-only route is not viable. The table shows how the counterfactual response trades off against factual fit:

| Variance penalty μ | Slope (x2) | Glucose NLL change |
|---|---|---|
| 30 | ≈ 0.18 | +0.0035 |
| 3 | ≈ 0.47 | +0.0073 |
| 0 | 0.69 | factual tilt sd of 460 mg/dL |

The head's input carries too little clean dose information to support a counterfactual shift without adding factual noise.

A precise dose signal exists only in u, the encoder output for the drug atoms (rate R² 0.93, slopes about 1). In each bundle the state's slots can only read u, and the latent z token that z_post computes from u and the context, through the prelude self-attention. So any clean relay to the head must pass through that attention. It is usable only if some attention head gives the u or z token a large and stable weight at insulin bundles.

To check, I'll make a copy of the hooked engine with a pre-step hook that captures the prelude inputs: the decayed slots, e, u and z. At insulin bundles I'll compute the prelude attention weights from each slot to each token, per head (mean and sd across rows).

- If a head puts a weight of about 0.5 or more on u or z with little spread, I'll build a value channel in that head (decode the rate, or the deviation, and write it into a fixed slot direction).
- If not, the weights are too variable, and I'll try a z_post channel or rethink.

```python
src=open('/work/rollout_h.py').read()
src=src.replace("def run_rollout_h(R,W,arms,n_paths=20,seed=0,t_hide=13.0,want_nll=False,chunk=None,store=None,store_h=False,hook_P=None,hook_draw=None,hook_state=None):",
 "def run_rollout_h2(R,W,arms,n_paths=20,seed=0,t_hide=13.0,want_nll=False,chunk=None,store=None,store_h=False,hook_P=None,hook_draw=None,hook_state=None,hook_pre=None,hook_z=None):")
old="        z=mlp(torch.cat([ctx,e,u],-1),W,'z_post')[:,:32]\n"
assert old in src
src=src.replace(old, old+"        if hook_z is not None: z=hook_z(k,act_n,z,ctx,e,u,dict(seq_si=seq_si,seq_arm=seq_arm))\n        if hook_pre is not None: hook_pre(k,act_n,hf,ctx,e,u,z,dict(seq_si=seq_si,seq_arm=seq_arm))\n")
open('/work/rollout_h2.py','w').write(src); exec(src)
A_=RD['A']; insS=np.where((A_.code.values==CODE['insulin'])&(A_.kind.values==0))[0]
sk2j={(int(A_.si.values[j]),int(A_.bidx.values[j])):j for j in insS}
attS=[]
def hpre(k,act_n,hf,ctx,e,u,z,c):
    s=c['seq_si'][act_n]; m=np.array([(int(x),k) in sk2j for x in s])&(c['seq_arm'][act_n]==0)
    if not m.any(): return
    mi=torch.as_tensor(np.where(m)[0])
    X=torch.cat([hf[mi]+W['core.slots'][None],e[mi][:,None],u[mi][:,None],F.linear(z[mi],W['z_in.weight'],W['z_in.bias'])[:,None]],1)
    xn=LN(X,W,'core.prelude.0.norm1'); q=F.linear(xn,W['core.prelude.0.attn.q.weight']); kv=F.linear(xn,W['core.prelude.0.attn.kv.weight']); k_=kv[...,:192]
    B=X.shape[0]; qh=q.reshape(B,7,4,48).transpose(1,2); kh=k_.reshape(B,7,4,48).transpose(1,2)
    a=((qh@kh.transpose(-1,-2))/math.sqrt(48)).softmax(-1)   # B x 4heads x 7q x 7k
    attS.append(a[:,:,:4,:].numpy())
_=run_rollout_h2(RD,W,[{}],n_paths=1,seed=0,t_hide=1e9,hook_pre=hpre)
AT=np.concatenate(attS); print('rows',AT.shape)
tok=['s0','s1','s2','s3','e','u','z']
for hh in range(4):
    mA=AT[:,hh].mean(0); sA=AT[:,hh].std(0)
    print('head',hh,' mean attn from slots (rows=slot q) to tokens',tok); print(np.round(mA,2)); print(' sd',np.round(sA.mean(0),2))
```