The flow decay rates fall exactly in the stated range. Dropping out_norm(h) or swapping the input order makes things much worse, so the overall structure is right. The remaining error is concentrated in a few z dimensions (9, 20 and 31, with mean error 0.2–0.3), not spread evenly. That pattern suggests the stored "posterior mean" may combine prior and posterior, for example precision-weighted, or apply some squashing.

Next, for those dimensions, I compare my raw posterior mean, the prior mean, both log-sds and the actual stored z. If actual ≈ the precision-weighted combination of prior and posterior, that confirms the hypothesis. If actual is a plain function of my posterior, such as a scaled or clipped version, that points to squashing instead.

```python
def zparts(o):
    bb=build_bundles(ev0,o); rows_=[]
    for k in range(1,len(bb)):
        out_=torch.tensor(X0[k-1,:192]); h_=torch.tensor(X0[k-1,192:960]).reshape(4,192)
        dt=(bb[k][0]-bb[k-1][0])*3600; hf=h_*torch.exp(-dt*torch.exp(fa))
        ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10]),'gap','gelu')
        at=bb[k][1]; ob=[a for a in at if a['ek']!=2]; ac=[a for a in at if a['ek']==2]
        e_=enc(torch.stack([emb(a,o) for a in ob]),'encode','gelu') if ob else torch.zeros(192)
        u_=enc(torch.stack([emb(a,o) for a in ac]),'encode_action','gelu') if ac else torch.zeros(192)
        q_=MLP(torch.cat([ctx,e_,u_]),'z_post','gelu'); p_=MLP(ctx,'z_prior','gelu')
        rows_.append((q_[:32].numpy(), q_[32:].numpy(), p_[:32].numpy(), p_[32:].numpy(), X0[k,960:]))
    return [np.array(t) for t in zip(*rows_)]
mq,lq,mp_,lp,za = zparts(UB)
for d in [9,20,31,4,23]:
    print('dim',d,'corr(q,act)',round(np.corrcoef(mq[:,d],za[:,d])[0,1],3),'slope',round(np.polyfit(mq[:,d],za[:,d],1)[0],3),
          'mean |q-act|',round(np.abs(mq[:,d]-za[:,d]).mean(),3),'logsd q med',round(np.median(lq[:,d]),2),'logsd p med',round(np.median(lp[:,d]),2))
wq=np.exp(-2*lq); wp=np.exp(-2*lp); comb=(mq*wq+mp_*wp)/(wq+wp)
print('err raw', round(np.abs(mq-za).mean(),4), 'err precision-comb', round(np.abs(comb-za).mean(),4))
print(np.round(np.c_[mq[:6,9],za[:6,9],mp_[:6,9],lq[:6,9],lp[:6,9]],3))
```