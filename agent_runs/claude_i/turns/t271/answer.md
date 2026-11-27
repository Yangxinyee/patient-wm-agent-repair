The prelude FFN has no dead units: the smallest activation SD on slot tokens is 0.05. More importantly, right after the prelude attention the slots already lose most of the dose detail. Rate R2 is 0.70 and policy-expected rate R2 is 0.90, but the deviation is only about 0.28–0.31 in every slot. The u token holds the rate linearly with R2 0.83, so attention transports it poorly.

The only place where both the context (expected rate) and the actual rate are cleanly available is z_post. So the path has to be:
1. z_post spare units compute a gated deviation into one z dim.
2. A redesigned z token carries it.
3. A rank-1 edit of the prelude value projection lets the slots read it strongly.

That only works if the slots give the z token a non-negligible and fairly stable attention weight.

Next I'll pin down the attention layout (head count and k/v split) by matching a manual computation to `attnw`. Then I'll measure the attention weights from slots to the e, u and z tokens at insulin bundles, and the norm of the z token. A z-token weight of at least about 0.02 with modest spread makes the value-edit route viable. A near-zero weight would also require a key edit.

```python
Wq = WT[P+'attn.q.weight']; Wkv = WT[P+'attn.kv.weight']; Wo = WT[P+'attn.out.weight']
def my_attn(xq, xkv, H, split='kv'):
    q = xq@Wq.T; kv = xkv@Wkv.T
    if split=='kv': k, v = kv[:,:192], kv[:,192:]
    else: kv2 = kv.reshape(len(xkv), H, 2, 192//H); k, v = kv2[:,:,0].reshape(len(xkv),192), kv2[:,:,1].reshape(len(xkv),192)
    dh = 192//H; qh = q.reshape(-1,H,dh).transpose(0,1); kh = k.reshape(-1,H,dh).transpose(0,1); vh = v.reshape(-1,H,dh).transpose(0,1)
    a = torch.softmax(qh@kh.transpose(1,2)/dh**0.5, -1); return (a@vh).transpose(0,1).reshape(-1,192)@Wo.T, a
k = ki[3]; hf = HH[k-1]*torch.exp(-dts[k]*torch.exp(fa))
X = torch.cat([hf+WT['core.slots'], Ee_[k][None], Uu_[k][None], Ztok[k][None]],0); Xn = lnorm(X, WT[P+'norm1.weight'], WT[P+'norm1.bias'])
ref = attnw(Xn[None,:4], Xn[None], P+'attn')[0]
for H in [1,2,3,4,6,8,12,16]:
    for sp in ['kv','int']:
        o,_ = my_attn(Xn[:4], Xn, H, sp); print(H, sp, '%.2e'%(o-ref).abs().max().item(), end=' | ')
print('\nztok norm med %.3f  ztok-centered norm med %.3f'%(Ztok.norm(dim=1).median(), (Ztok-Ztok.mean(1,keepdim=True)).norm(dim=1).median()))
```