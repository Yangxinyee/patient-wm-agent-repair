The notebook append did not run: the script stopped earlier because `load()` returns a torch tensor, not a numpy array. The useful results did come through. z-dims 18, 28 and 27 vary least (sd 0.05–0.06). Every z_in column has norm about 1.6, so hijacking dim 18 for a new signal would disturb the factual z token by only about 0.08 in norm.

head2.py recalled my head reproduction: `value_mean(ctx, codes, lastv, lastage_s, ...)` with the head's last-value register.

Next I will test leverage. At drug-A bundles I will push the z token along each of the 192 token directions and see how much the next glucose prediction moves. First I need the names and shapes of the per-bundle arrays in the kernel (states, e/u, dt, times, registers). This run lists them, computes the typical z-token norm, and redoes the notebook append. I expect arrays of length 5156 for HHt/OUTt/Ee/Uu/dts/tmin plus register arrays. If those are missing, I will rebuild them from fwd_3.

```python
import torch
arr=[]
for k,v in list(globals().items()):
    if k.startswith('_'): continue
    if isinstance(v,(np.ndarray,torch.Tensor)) and v.ndim>0: arr.append(f"{k}{tuple(v.shape)}")
    elif isinstance(v,(list,dict)) and len(v)>50: arr.append(f"{k}[{type(v).__name__}:{len(v)}]")
print(' '.join(arr)[:2500])
Wz=load('z_in.weight'); Wz=Wz.numpy() if hasattr(Wz,'numpy') else Wz; bz=np.asarray(load('z_in.bias'))
print('z token norm mean', round(float(np.linalg.norm(np.asarray(ZZt)@Wz.T+bz,axis=1).mean()),2))
with open('/work/notes.md','a') as f: f.write("""
## Session 6 (restart after reply 220)
- ARCH recap: step = flow decay; ctx; z_post mean (state update only); prelude tokens [h+slots, e, u, z_in(z)] -> E=LN(slots);
  h <- a_bar*h + softplus(delta_in)*b_in(E) + blocks increments; y=coda(c_out(h)); out=LN(mean slots). Heads read
  view = out + LN(mean h_flowed) + gap + z_head(z_prior). Value = softmax 17 bins (attr_heads.value_bin MLP + transition table).
  -> drug info reaches value predictions ONLY via prelude tokens u,z. z_post logvar outputs unused at eval.
- JI/JN (z-dim Jacobians on downstream preds): tiny (~0.03/unit z, z sd ~0.17), sign flips -> adding along existing z dirs is weak.
- least-used z dims: 18 (sd .049), 28, 27 (z_in col norms all ~1.6).
- PLAN: hijack z dim 18: z_post dead units compute gated deviation (rate(u)-pibar(ctx)); redesign z_in column 18.
  One-step injection (shift next reading) + rollout propagation via drawn readings; calibrate scale with dose_edit.
""")
```