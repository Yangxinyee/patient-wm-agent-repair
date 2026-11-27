Run 93 shows that a fitted constant added to ctx lowers the error only to about 0.41, and it is not aligned with gap(x) for any x. So the mismatch is not just a missing constant gap vector.

One documented phrase I took literally may be the cause: "u_k (zero if the bundle has none)". The model may instead encode an empty action set. That gives a non-zero constant, the normalised MLP of the size term, in the u input of z_post. I'll test that directly, both with and without the gap term and the time flow.

- **If it works:** the error drops sharply, and the rebuild is nearly complete.
- **If it fails:** I'll pivot to designing the edit and verifying it with the real model.

```python
def setenc_empty(p):
    s = lin_(torch.tensor([math.log(1+0)]), p+'.size')
    return ln_(mlp_(s, p+'.mlp2', 'gelu'), p+'.norm')
U_empty = setenc_empty('encode_action')
GG_s10 = torch.stack([G_in(d_, 's10') for d_ in dts_])
for nm_, ctx_ in [('out+LNhf', O_ + HF_), ('out+LNhf+gap', O_ + HF_ + GG_s10), ('out+LNh', O_ + HN_), ('out+LNh+gap', O_ + HN_ + GG_s10)]:
    zu = mlp_(torch.cat([ctx_, EE_, U_empty[None].expand_as(EE_)], 1), 'z_post', 'gelu')[:, :32]
    print(nm_, 'u=enc(empty):', round(float(torch.median(torch.norm(zu-ZT_, dim=1)/torch.norm(ZT_, dim=1))),4))
```