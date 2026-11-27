Run 86 shows that no reweighting of the ctx pieces helps: the best combination is still at 0.54. So ctx is probably built as documented. The remaining error must sit in a stage applied to both inputs: the activation inside z_post, or the activations in the set encoders. I had assumed GELU everywhere, and an MLP using SiLU or tanh would give exactly this kind of partial agreement.

I'll test the activation of z_post on the cached inputs. Then I'll test the encoder activations by recomputing e with each candidate, holding z_post fixed at the best one found. A large drop in error for one activation would settle it.

```python
ACT2 = {'gelu': Fnn.gelu, 'gelu_tanh': lambda x: Fnn.gelu(x, approximate='tanh'), 'relu': Fnn.relu, 'silu': Fnn.silu,
        'tanh': torch.tanh, 'elu': Fnn.elu, 'leaky': Fnn.leaky_relu, 'softplus': Fnn.softplus}
def mlp_a(x, p, af): return lin_(af(lin_(x, p+'.0')), p+'.2')
ctx_b = O_ + HF_
rz = []
for an_, af_ in ACT2.items():
    zh = mlp_a(torch.cat([ctx_b, EE_, torch.zeros_like(EE_)], 1), 'z_post', af_)[:, :32]
    rz.append((round(float(torch.median(torch.norm(zh-ZT_, dim=1)/torch.norm(ZT_, dim=1))),4), an_))
rz.sort(); print('z_post act:', rz)
best_zact = ACT2[rz[0][1]]
wa_list = []
for s_ in list(bundles_B.keys())[:20]:
    bl_ = bundles_B[s_]
    for k_ in range(1, len(bl_)):
        atoms = bl_[k_][1]
        wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
        if aa or not wa: continue
        wa_list.append(wa)
        if len(wa_list) >= 150: break
    if len(wa_list) >= 150: break
def setenc_a(vecs, p, af):
    n = len(vecs); m1 = mlp_a(torch.stack(vecs), p+'.mlp1', af)
    return ln_(mlp_a(m1.sum(0)/math.sqrt(n) + lin_(torch.tensor([math.log(1+n)]), p+'.size'), p+'.mlp2', af), p+'.norm')
re_ = []
for an_, af_ in ACT2.items():
    EEa = torch.stack([setenc_a([atom_vec(a, optA) for a in wa], 'encode', af_) for wa in wa_list])
    zh = mlp_a(torch.cat([ctx_b, EEa, torch.zeros_like(EEa)], 1), 'z_post', best_zact)[:, :32]
    re_.append((round(float(torch.median(torch.norm(zh-ZT_, dim=1)/torch.norm(ZT_, dim=1))),4), an_))
re_.sort(); print('encoder act:', re_)
```