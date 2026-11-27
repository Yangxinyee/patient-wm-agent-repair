The inventory shows that I already have the third forward pass's states (`X3`, `HHt`, `OUTt`, `ZZt`), the recomputed contexts (`ctx2`), the encodings (`Ee`, `Uu`) and drug-bundle arrays (`drugb`, `ctxD`, `UD`, `insr_`, `ner_`, `pb_`, `gdec`). There are also arrays from a Jacobian computation done just before the restart: `JG` (300 × 32 × 4) with its mean `mJ` and spread `sJ`. They probably hold the derivative of the next predicted readings with respect to each z dimension at 300 bundles.

I can't see the code that produced them. So in this run I will:
- look at them only briefly;
- recompute the dead `z_post` units myself from `ctx2`, `Ee` and `Uu`;
- save my own code in a file so the work survives restarts.

The analysis rests on two things being true. First, the dead units must show near-zero activation spread with strongly negative pre-activations, so they can be repurposed. Second, `mJ` must show z directions that move glucose much more than MAP. If no such directions exist, I will have to redesign around a different readout.

```python
import torch.nn.functional as Fnn
print('samp range', samp.min(), samp.max(), '| drugb range', drugb.min(), drugb.max())
print('mJ (mean dPred/dz, 32x4):'); print(np.round(mJ,3).T)
print('sJ:'); print(np.round(sJ,3).T)
x = torch.cat([ctx2, Ee, Uu], -1).double()
pre_z = x @ WT['z_post.0.weight'].double().T + WT['z_post.0.bias'].double()
actz = Fnn.gelu(pre_z)
sdz = actz.std(0).numpy()
dead = np.where(sdz < 0.01)[0]
print('n dead', len(dead), dead.tolist())
print('dead pre-act max/mean', np.round(pre_z[:,dead].max(0).values.numpy()[:10],2), np.round(pre_z[:,dead].mean(0).numpy()[:10],2))
w2 = WT['z_post.2.weight'].double().numpy()
print('out-weight norm of dead units (mean rows/ls rows):', np.round(np.linalg.norm(w2[:32][:,dead],axis=0)[:8],3), np.round(np.linalg.norm(w2[32:][:,dead],axis=0)[:8],3))
```