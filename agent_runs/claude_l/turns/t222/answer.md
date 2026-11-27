The prelude layer is a pre-norm 4-head attention (48 dimensions per head): queries from `attn.q`, keys and values stacked in `attn.kv`, then `attn.out`, followed by an FFN.

The step response suggests a better design than tie-splitting, a "swap":
- **Retire u:** shift every action encoding u by a constant b′ through the `encode_action.norm.bias`. Choose b′ so that u's attention scores in all heads fall far below the other tokens', and u is never attended.
- **z takes u's place:** z_post subtracts b′ and outputs an exact copy of the original u, plus the gated residual injection. The z token then holds u's role in every head, for both drug and reading bundles.
- **Result:** with no injection the model is unchanged, apart from a ~0.1% copy error. The injection acts through z's value smoothly and in both signs, instead of through a winner-take-all tie.

To design b′ I need the attention scores. Next I'll print my replica's attention code (`bmha`, `bself`) so I can compute per-head scores with the correct key/value layout. Then I'll measure at sampled bundles how far the winning token's score exceeds the rest, and how a constant b′ moves u's scores.

```python
src2=open('/work/wm2.py').read()
for nm in ['def bmha','def bself','def bln']:
    i=src2.find(nm); j=src2.find('\ndef ',i+5); print(src2[i:j if j>0 else i+700][:700]); print('--')
```