`W2` edited only the prelude output projection, by amplifying a decoder (norm 223) on head 2's existing dose encoding about 650-fold. Run 252 showed that encoding is category-like, so that attempt could not respond linearly to raised doses.

I've now decided on the repair: a genuinely linear dose channel. The encoder pools atoms as `LN(mlp2(Σ mlp1(atom)/√n + size(log1p n)))`. Its only linear dose input is the numeric z_rate column. So the channel will be:
- **mlp1:** a pair of linear-regime GELU units per drug, gated on the drug code. Their difference carries α·rate per START atom, and is 0 when there is no infusion.
- **mlp2:** a unit that passes it into a quiet direction of the encoder output, kept small so the LayerNorm stays linear.
- **Prelude:** a private dimension in head 2's value (attention to u ≈ 1), mapped by `attn.out` into the slots.
- **Head:** read from there.

I'll build and test it stage by stage on my exact local model. First I must check that my rollout and encoding functions recompute the encoder from the edited weight dict, rather than reusing cached u vectors. I also need how atoms store rate and dose, and whether the rollout returns a factual NLL I can use locally. This run only reads code and data structures.

```python
A_=Rx['A']; print('A cols', A_.columns.tolist())
src=open('/work/rollout_h2.py').read().split('\n')
for i,l in enumerate(src):
    if any(k in l for k in ['encode','arm_embed','xa','set_encode','def ','nll']): print(i, l.strip()[:140])
print('res_w keys', list(res_w.keys()))
print({k:np.shape(v) for k,v in Dw.items()})
```