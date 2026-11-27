The architecture text clarifies where dose information can travel. The doses enter only through the action vector u_k, which feeds z_post and the prelude. The memory M stores only measurement-bundle vectors. Since infusion END atoms go to e, M does hold past doses indirectly. The heads see only slot means of h, through `out` and the view.

So any lasting dose signal must live in h, or be read from M by the blocks' cross-attention. That explains why a head-only readout of the dose innovation was weak.

For a principled edit, I need a working replica of one core step that can be checked against the tool's saved states, using teacher-forced previous states. The earlier attempt failed (error 1.79), before I had re-read these details.

Next, I will list all core, latent and head tensors with their shapes, plus the model config: number of attention heads and memory size.

How this decides the next step:
- **Replication goes ahead:** the shapes are consistent with a standard pre-norm layer (q / kv / out projections, a 4x FFN, three norms) and the config gives a head count. Then I try a one-step replica next.
- **Approach changes:** unexpected tensors appear, for example norms on keys or gating parameters.

```python
cfg = VOCAB.get('config', {})
print('config:', cfg)
for n in NAMES:
    if n.startswith('core.') and not n.startswith('core.blocks.0.1') :
        print(n, SHAPES[n])
print('---- non-core, non-embed')
for n in NAMES:
    if not n.startswith('core.') and not n.startswith('embed.'):
        print(n, SHAPES[n])
```