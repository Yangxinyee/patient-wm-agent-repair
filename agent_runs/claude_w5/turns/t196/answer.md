The expected-dose term is almost perfectly linear in the prelude input. Held-out R² from n3 is 0.995 for insulin and 0.97 for norepinephrine, and the readout ignores the dose variants as intended.

So:
- the centring part is easy from n3;
- the dose itself is easy from u;
- the only missing link is a clean path for the dose from the u token into the slots.

Next, stage 1. I will collect, at every bundle and dose variant, the norm1 outputs of all seven prelude tokens and head 1's attention weights. Head 1 puts about 1.0 on u at drug bundles and about 0.95 on the empty-drug token u0 elsewhere.

Then I will fit readouts q_I and q_N on the u token's norm1 output with these constraints:
- they give the insulin or norepinephrine rate;
- they are exactly 0 on u0;
- they are close to 0 on every other token.

Finally I check the attention-weighted signal head 1 would deliver: about equal to the rate at drug bundles, about 0 elsewhere.

```python
WKV=W64['core.prelude.0.attn.kv.weight']; WQ=W64['core.prelude.0.attn.q.weight']
def tok_n1(r,at):
    e,u=enc2(at); hf=r['h0']*torch.exp(-r['dt']*torch.exp(W64['core.flow.a']))
    ctx=r['out0']+ln_(hf.mean(0),'core.out_norm')+mlp_(torch.tensor([math.log(max(r['dt'],1e-9))/10],dtype=torch.float64),'gap')
    zz=mlp_(torch.cat([ctx,e,u]),'z_post')[:32]
    Tk=torch.cat([hf+W64['core.slots'],e[None],u[None],lin_(zz,'z_in')[None]]); n1=ln_(Tk,'core.prelude.0.norm1')
    q=(n1@WQ.T)[:4].reshape(4,4,48); k=(n1@WKV[:192].T).reshape(7,4,48)
    A=torch.softmax(torch.einsum('shd,thd->hst',q,k)/math.sqrt(48),-1)   # heads x slots x tokens
    return n1,A
u0n1=None; TOK=[]   # (n1 7x192, A, ri, rn, isdrug, fit)
for d in D:
    s,k=d['sid'],d['k']; r=RK[s][k]; at=bundles_B[s][k][1]
    for sfx,fi,fn_ in (VK if isdrug(d) else [('',1,1)]):
        atv=at if sfx=='' else scale(at,*VN[sfx[1:]])
        n1,A=tok_n1(r,atv); TOK.append((n1.float(),A.float(),fi*d['ri'],fn_*d['rn'],isdrug(d),d['sid'] in FIT))
        if not isdrug(d) and u0n1 is None: u0n1=n1[5].clone()
print('tokens',len(TOK),'head1 attn to u: drug %.3f nondrug %.3f'%(np.mean([float(t[1][1,:,5].mean()) for t in TOK if t[4]]),np.mean([float(t[1][1,:,5].mean()) for t in TOK if not t[4]])))
print('u0 n1 identical across nondrug:',max(float((t[0][5]-u0n1.float()).abs().max()) for t in TOK if not t[4]))
```