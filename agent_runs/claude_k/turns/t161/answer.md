The writer edit is right offline:
- **Units.** Units 55 and 109 are effectively dead (largest |activation| ≤ 0.001 on drug and reading bundles), so repurposing them costs nothing.
- **Insulin channel.** z₁₈ now moves by 0.290 per unit/h of insulin (correlation 0.985). That is slightly below the 0.3 design because the ridge decoder shrinks a little.
- **Norepinephrine channel.** z₂₈ moves by 1.83 per mcg/kg/min (correlation 0.947).
- **Everything else.** All other z dimensions are unchanged, and reading bundles are unchanged to within 0.0001.

The save failed only because torch wasn't imported, so I'll import it, save the four z_post tensors, and log the design in the notebook. My next reply will then request original and edited states on the same 99 stays, to see where this injection lands in q at the following readings.

```python
import torch
for n_,a_ in [('z_post.0.weight',W0n),('z_post.0.bias',b0n),('z_post.2.weight',W2n),('z_post.2.bias',b2n)]: save_edit(n_, torch.tensor(a_,dtype=torch.float32))
print('edited', edited(), 'units', j_ins, j_ne)
np.save('/work/p_ins.npy',p_ins); np.save('/work/p_ne.npy',p_ne)
open('/work/notes.md','a').write("""- run140: q rebuild exact (view_from(prev_row=S2 row k, dt), z=prior mean) == Q99. u_k rebuild: setenc_my(drug_atoms(i,t,'both',mult)) (empty->0); z_post recon err .05 (ctx-side).
- run144-145: linear decoders of rate from u_k (no intercept, ridge a=.01, trained on dose-edit-augmented encodings): INS |p|379 ratios ~1 at x0..x3; NE |p|142. saved /work/p_ins.npy p_ne.npy
- run147: WRITER EDIT SAVED (z_post.0/.2 w,b): repurposed dead z_post units 55 (ins) & 109 (ne): pre=5+p.u, out col -> z18 (lam .3/unit/h) / z28 (lam 2 per mcg/kg/min), bias -lam*gelu(5). offline: dz18=.29*rate, dz28=1.83*rate, others 0.
  NEXT: fwd states both on sel_states (99) -> measure delta-q at readings; then value-head paired-unit readout.
""")
```