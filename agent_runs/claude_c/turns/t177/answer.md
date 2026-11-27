Head 0 ×4 raises factual NLL to 1.64132, +0.0067 over the original. That is over the 0.005 budget, so head 0 also carries reading information.

A key fact from the removal test: the action token's whole influence is about 1/13 of the true effect. Input-side edits therefore can't reach the truth however rate-sensitive they make the token. Raising the token's share needs an attention-level change.

Since the state direction is very stable, the slot queries are nearly fixed. Sharpening head 0's attention (scaling its query rows ×3) could make it focus on the action token at drug bundles, if that token has the top logit there. In that case insulin readouts should rise well beyond the ×4 output scaling's −2.53 / +1.67, and I'll check factual cost next. If they fall or barely move, the action token isn't head 0's preferred token.

```python
clear_ops()
for r in range(48):
    e = np.zeros(192); e[r] = 1.0; rank1('core.prelude.0.attn.q.weight', e, 2.0, side='out')
print(len(ops()))
with open('/work/notes.md','a') as f:
    f.write("- Pass 22: head0 x4 factual 1.64132 (+0.0067) -> too costly. Next: sharpen head0 attention (q rows 0-47 x3).\n")
```