The encoders and z posterior are now reproduced exactly on every bundle type. The unit table is mmHg 2, units 3, mg 4, and mg/dL 5 for lab glucose only; finger glucose has no unit embedding.

Next is the core step itself: prelude, recurrent blocks with cross-attention to memory, gated update, coda and output norm. I'll check it against the stored out and h after each bundle, using the exact e_k, u_k and stored z_k. First I need the layer names inside a recurrent block, to see how its self-attention, cross-attention, norms and FFN are laid out. I'll also note this progress in the notebook.

```python
atom_e=atom_e_full(4,3)
with open('/work/notes.md','a') as fh:
    fh.write("- Runs 70-74: z_post replica EXACT on all bundle types. Units: mmHg->2, units(insulin dose)->3, mg(NE dose)->4, glucose_lab mg/dL->5, glucose_finger NO unit. Bins side='left'. Use atom_e_full(4,3).\n")
print([n.replace('core.blocks.0.','') + ':' + str(tuple(SHAPES[n])) for n in NAMES if n.startswith('core.blocks.0.0.')])
print([n for n in NAMES if n.startswith('core.blocks.')][:3], len([n for n in NAMES if n.startswith('core.blocks.0.1.')]))
print([n+':'+str(tuple(SHAPES[n])) for n in NAMES if n.startswith('core.age') or n.startswith('core.coda.0.attn') or n.startswith('core.prelude.0.attn')])
```