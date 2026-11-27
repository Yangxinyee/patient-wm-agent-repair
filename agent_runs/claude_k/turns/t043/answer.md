The tensor list shows where a correction could go:

- **Value head (`attr_heads.value_bin`):** a two-layer MLP, Linear 192→192, an activation, then Linear 192→17. It is shared by all four reading codes.
- **`embed_transition.value_bin`:** a bias table of shape (13 codes × 17 last-bins) → 17 logits.
- **`z_post`:** takes [ctx, e_k, u_k] (576 inputs).
- **Dose path:** the dose reaches the heads only through `encode_action` → u_k → `z_post` and the prelude → state → view → q.

My plan is to add a code-gated, roughly linear term to the value head's hidden layer. It would read a direction in q that encodes recent dose relative to what the readings imply. Gating would make insulin affect only glucose codes and norepinephrine only MAP codes.

To design this I need:
- the full code list, with ids for the gating;
- the glucose and MAP bin centres, to turn a mean shift into logits;
- the rate and dose normalisation for the two drugs;
- which activation the head MLPs use.

This run gathers that information and saves the architecture facts to my notebook.

```python
import json, re
codes = json.loads(json.dumps(VOCAB['codes']))
for c in codes: print(c)
print()
for e in VOCAB['numeric']:
    print(e)
print()
for k2, v2 in VOCAB['value_bin_centres'].items(): print(k2, v2)
arch = open('/model/ARCHITECTURE.md').read()
for kw in ['gelu','GELU','relu','ReLU','silu','tanh','activation','LayerNorm','RMS']:
    if kw in arch: print('ARCH mentions', kw)
with open('/work/notes.md','a') as fh:
    fh.write("\n## Architecture facts (run 40-41)\n- value_bin head: Linear192->192, act, Linear192->17 (shared by all reading codes); embed_transition.value_bin (13*17 x 17) bias by code x last bin.\n- z_post.0 (192x576) on [ctx,e,u]; dose reaches heads only via u_k -> z_post/prelude -> state -> view -> q.\n- Plan: code-gated linear correction term in value head hidden layer, reading a dose direction in q.\n")
```