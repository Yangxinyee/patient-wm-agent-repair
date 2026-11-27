Two facts settle the u_k structure:

1. **No drugs.** Every bundle without a drug has the same constant u_k. This is the empty-set encoding, roughly LN(mlp2(size.bias)), not a literal zero vector.
2. **Insulin only.** u_k depends only on the insulin rate, not on n_atoms or co-occurring END atoms. So u_k is a deterministic function of the drug rates in the bundle: U_I(r), U_N(r) and U_IN(r_I, r_N).

That makes a CPU replica of `encode_action` feasible. I can check it against the probe's 20 exact coordinates in stages:
- the empty set, which tests mlp2, size, LayerNorm and GELU;
- single insulin atoms, which tests the atom embedding;
- then NE atoms and combinations.

The empty-set check also needs no guessing about atom fields. Next I'll print the vocab entries needed for the atom embedding: codes, insulin and NE normalisation, rate and dose bin edges and categorical indices. Then I'll test the empty-set replica straight away. If LN(mlp2(size.bias)) under GELU matches the probe's constant vector, the encoder's back half is confirmed.

```python
V = VOCAB
print(list(V.keys()))
print('codes', V['codes'] if not isinstance(V['codes'], dict) else list(V['codes'].items())[:13])
for k in V:
    if k not in ('codes',):
        s = json.dumps(V[k]); print(k, s[:300])
# empty-set replica check
def gelu(x): return torch.nn.functional.gelu(x)
def LNf(x, w, b): return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, eps=1e-5)
def mlp_(pfx, x): return load(pfx+'.2.weight') @ gelu(load(pfx+'.0.weight') @ x + load(pfx+'.0.bias')) + load(pfx+'.2.bias')
def enc_set(pfx, atoms):  # atoms: (n,192) tensor
    n = atoms.shape[0]
    s = (torch.stack([mlp_(pfx+'.mlp1', a) for a in atoms]).sum(0)/np.sqrt(n)) if n>0 else torch.zeros(192)
    s = s + load(pfx+'.size.weight')[:,0]*np.log1p(n) + load(pfx+'.size.bias')
    return LNf(mlp_(pfx+'.mlp2', s), load(pfx+'.norm.weight'), load(pfx+'.norm.bias'))
for nm in ['z_post.0.weight','z_post.0.bias','z_post.2.weight','z_post.2.bias','z_in.weight']: pass
u_empty = enc_set('encode_action', torch.zeros(0,192))
print('replica empty u at probe coords:', u_empty[u_c][:8].numpy().round(3))
print('probe   empty u              :', U[nod][0][:8].round(3))
```