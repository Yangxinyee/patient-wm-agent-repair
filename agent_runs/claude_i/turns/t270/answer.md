`attnw` takes batched (query, key/value) inputs and returns the attention output, here (1, 4, 192). Together with `lnorm` and `ffnw`, I can rebuild the prelude's slot internals from the fwd_3 states:
- tokens are [flowed h + slots, e, u, z_in(z)];
- each slot gets norm1, attention and its residual, then norm3 and the FFN.

Next I'll compute, for all fwd_3 bundles, each slot's FFN input (`norm3` of the post-attention residual) and the 768 FFN hidden activations. That answers two questions:
1. Are there FFN units that are dead on slot tokens? These are free to repurpose.
2. At insulin bundles, can the dose deviation (rate − policy-expected rate) be read linearly from the FFN input? Ridge CV R2 for rate, expected rate and deviation.

A deviation R2 well above the final state's 0.24 makes an FFN-based gated deviation unit feasible. A low R2 means attention discards the u token's rate before the FFN.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict as cvp
gelu = torch.nn.functional.gelu
P = 'core.prelude.0.'; NBn = len(INFO)
sid_ = np.array([d['sid'] for d in INFO])
Ztok = (ZZt.float() if torch.is_tensor(ZZt) else torch.tensor(ZZt)).float() @ WT['z_in.weight'].T + WT['z_in.bias']
Ee_ = torch.as_tensor(Ee).float(); Uu_ = torch.as_tensor(Uu).float(); HH = torch.as_tensor(HHt).float()
F3 = torch.zeros(NBn,4,192); X1s = torch.zeros(NBn,4,192)
for k in range(NBn):
    if k==0 or sid_[k]!=sid_[k-1]: hf = torch.zeros(4,192)
    else: hf = HH[k-1]*torch.exp(-dts[k]*torch.exp(fa))
    X = torch.cat([hf+WT['core.slots'], Ee_[k][None], Uu_[k][None], Ztok[k][None]],0)
    Xn = lnorm(X, WT[P+'norm1.weight'], WT[P+'norm1.bias'])
    X1 = X[:4] + attnw(Xn[None,:4], Xn[None], P+'attn')[0]
    X1s[k] = X1; F3[k] = lnorm(X1, WT[P+'norm3.weight'], WT[P+'norm3.bias'])
ACT = gelu(F3 @ WT[P+'ffn.0.weight'].T + WT[P+'ffn.0.bias'])
sd = ACT.reshape(-1,768).std(0); w2n = WT[P+'ffn.2.weight'].norm(dim=0)
print('dead units (sd<0.01):', int((sd<0.01).sum()), ' sd<0.03:', int((sd<0.03).sum()), ' min sd %.4f'%sd.min())
ki = np.array([k for k in range(1,NBn) if 'insulin' in INFO[k] and sid_[k]==sid_[k-1]])
G = np.array([INFO[k-1].get('glucose_finger', INFO[k-1].get('glucose_lab', np.nan)) for k in ki]); ok = ~np.isnan(G)
ki, G = ki[ok], G[ok]; rate = np.array([INFO[k]['insulin'] for k in ki]); pb = np.asarray(pibar(G)).ravel(); dev = rate - pb
for sl in range(4):
    Xf = F3[ki, sl].numpy()
    r2 = [1 - ((cvp(RidgeCV(alphas=np.logspace(-2,3,8)), Xf, y, cv=5) - y)**2).mean()/y.var() for y in (rate, pb, dev)]
    print('slot', sl, 'R2 rate %.2f pibar %.2f dev %.2f'%tuple(r2))
```