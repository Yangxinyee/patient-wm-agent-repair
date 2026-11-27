Amplifying the drug numerics 4× did essentially nothing:
- insulin ×2: −2.01 → −2.06;
- insulin ×½: +1.16 → +1.52;
- NE ×2: +0.37 → −0.06.

So the numeric inputs are not a lever. The model's whole dose response is weak because little flows from the drug and z tokens into the state, and both enter through the prelude attention, where layer norm caps what an added token vector can do.

The remaining strong lever is the prelude's value projection, which acts after the token layer norm and is linear. Here is the plan:
- **Write the signal:** keep my gated z-posterior units, which write the insulin deviation into z dim 18 and the NE rate into dim 28.
- **Choose a quiet direction:** make z_in columns 18 and 28 move the z token along a direction that no other token (slots, e, u, z) normally uses.
- **Amplify it:** add a rank-1 term to the prelude value projection that maps that quiet direction to the value change a lower glucose reading (or a higher MAP reading) produces through the e token. Its gain can be large without saturating, because the side effects scale only with the tiny variance other tokens have along the quiet direction.

To write this I need the prelude's exact tensors: attention input/output projections, layer norms, heads, slot embeddings and the flow parameter. So I'll undo the useless numeric edit and list those tensors.

```python
drop_edit('embed.numeric.weight'); print('edited',edited())
with open('/work/notes.md','a') as f_: f_.write("- fwd4: numeric x4 amplification -> no effect (ins x2 -2.06, NE x2 -0.06). Dropped. Next: rank-1 edit of prelude value proj on a quiet z-token direction, fed by gated z_post dev units (edit_parts.npz).\n")
pn=[f"{n}{tuple(SHAPES[n])}" for n in NAMES if n.startswith('core.prelude') or n in ('core.slots','core.flow.a','core.a_in','core.delta_in')]
print(' '.join(pn))
```