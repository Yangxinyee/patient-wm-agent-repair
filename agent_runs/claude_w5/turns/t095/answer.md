Run 90 rules out the gap-placement variants: none comes near an exact match. I have spent many runs on this rebuild without converging. The one stage I haven't questioned is the raw atom vector itself: it may be normalised or rescaled before the set encoder, for example by a LayerNorm without learned weights or by a fixed factor. That would distort every reading's embedding in the systematic way I'm seeing.

I'll test those transforms on the cached measurement-only bundles. If none of them helps, I'll stop chasing an exact rebuild. Instead I'll design the edit so that it can be checked with the real model through the forward tool, especially its states requests, rather than relying on a local copy.

```python
def atom_tf(v, mode):
    if mode == 'none': return v
    if mode == 'ln': return Fnn.layer_norm(v, (192,), None, None, 1e-5)
    if mode == 'rms': return v / torch.sqrt((v*v).mean() + 1e-6)
    if mode == 'x2': return v*2
    if mode == 'x0.5': return v*0.5
    if mode == 'sqrtd': return v*math.sqrt(192)
    if mode == 'div_sqrtd': return v/math.sqrt(192)
res_tf = []
for mode in ['none','ln','rms','x2','x0.5','sqrtd','div_sqrtd']:
    EEm = torch.stack([setenc_a([atom_tf(atom_vec(a, optA), mode) for a in wa], 'encode', Fnn.gelu) for wa in wa_list])
    res_tf.append((round(batch_err(ctx_b if False else O_ + HF_), 4) if False else round(float(torch.median(torch.norm(zfun(O_ + HF_, EEm)[0]-ZT_, dim=1)/torch.norm(ZT_, dim=1))),4), mode))
print(res_tf)
```