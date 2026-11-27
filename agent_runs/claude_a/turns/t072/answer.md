Run 67 gave the full core layout:
- The prelude and coda are single pre-norm transformer layers (self-attention + FFN).
- The two recurrent blocks add a cross-attention to the memory M, with age-encoded keys.
- `c_out` and `b_in` are bias-free linear maps; the attention projections have no biases.
- The state update has the linear-recurrence form a_bar·h + softplus(δ)·b_in(E) + block residual, followed by the coda.

I have no core-step replica yet: no such function exists in the kernel. I do have the true per-bundle states (`HS`, `OUT`, `ZP`), the encoder outputs (`eALL`, `uALL`) and the ctx helper. That lets me validate a replica one step at a time, from h_{k-1}, e_k, u_k, z_k and M to h_k and out_k.

To write it I need the attention head count and the memory and time conventions. So next I'll print `VOCAB['config']`, the metadata in tensors.json, the bundle tables (`SB`, `MB`, `B`) and the range of `flow.a`. The range of exp(−flow.a) will confirm the time unit: about 600 to 6e7 would mean seconds.

```python
print("config:", VOCAB['config'])
k0 = 'core.blocks.0.0.cross.kv.weight'
print("TJ entry:", TJ.get(k0) if isinstance(TJ, dict) else None)
print("TJ keys sample:", list(TJ.items())[:2])
for nm_, df_ in [('SB',SB),('MB',MB),('B',B),('st',st)]:
    print(nm_, df_.shape, list(df_.columns)); print(df_.head(3).to_string())
fa_ = load('core.flow.a'); print("exp(-flow.a) range s:", float(torch.exp(-fa_).min()), float(torch.exp(-fa_).max()))
print("a_in", load('core.a_in')[:5], "delta_in", load('core.delta_in')[:5])
print("S keys:", list(S.keys()), {k: S[k].shape for k in S.keys()})
print("e_empty norm", float(e_empty.norm()), "u_empty norm", float(u_empty.norm()), "uALL zero rows", int((uALL.abs().sum(1)==0).sum()))
```