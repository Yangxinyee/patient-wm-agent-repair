The second failure shows that the cached losses have an extra sampling dimension, so they cannot be indexed as per-reading losses. No candidate has been scored or installed. I’ll verify the evaluator’s four-draw averaging convention and make the scalar scorer match it. Near-exact agreement on unchanged weights is required before the factual-loss frontier is trustworthy.

```python
print("=== CACHED OUTPUT SHAPES ===")
for label210, mc210 in (("train", wm_train145), ("tuning", wm_tune145)):
    print(label210)
    for k210 in ("inputs", "probs", "mean", "nll", "target"):
        x210 = mc210[k210]
        print(" ", k210, type(x210).__name__, getattr(x210, "shape", None))

print("\n=== LOSS-AVERAGING IMPLEMENTATION ===")
ins210 = list(dis.get_instructions(head177_eval))
start210 = next(i for i, x in enumerate(ins210)
                if x.opname == "LOAD_ATTR" and x.argval == "gather")
print(" | ".join(f"{x.opname} {x.argrepr}"
                 for x in ins210[start210:start210+75]))

def scalar_eval208(mc, baseline, coef, mask, centres):
    cc = torch.as_tensor(coef, dtype=torch.float32)
    tau = F.linear(mc["inputs"], cc[:192], cc[-1])
    tau = tau * torch.as_tensor(mask, dtype=torch.float32)[None, :]
    pp = torch.softmax(
        baseline["logits"] + tau[..., None]*v208, dim=-1)
    target = torch.as_tensor(mc.get("target", wm_train145["target"]),
                             dtype=torch.long)
    pt = pp.gather(-1, target[None, :, None].expand(pp.shape[0], -1, 1))
    pt = pt.squeeze(-1).reshape(-1, 4, len(target)).mean(1)
    nll = -pt.clamp_min(1e-30).log().mean(0)
    mean = (pp.mean(0) * centres).sum(-1)
    return nll.numpy(), mean.numpy()

zero210 = np.zeros(193)
zn210, zm210 = scalar_eval208(
    wm_train145, bt208, zero210, mapmask208, cent208)
ztn210, ztm210 = scalar_eval208(
    wm_tune145, bu208, zero210, tunemap208, tcent208)
print("\nZero-correction agreement:")
print("train max NLL/mean error:",
      float(np.max(np.abs(zn210-bt208["nll"]))),
      float(np.max(np.abs(zm210-bt208["mean"]))))
print("tuning max NLL/mean error:",
      float(np.max(np.abs(ztn210-bu208["nll"]))),
      float(np.max(np.abs(ztm210-bu208["mean"]))))
assert np.max(np.abs(zn210-bt208["nll"])) < 1e-5
assert np.max(np.abs(ztn210-bu208["nll"])) < 1e-5

orig_train209 = head177_eval(wm_train145, original206)["nll"]
orig_tune209 = head177_eval(wm_tune145, original206)["nll"]
print("Original per-reading reference shapes:",
      orig_train209.shape, orig_tune209.shape)
print("Calibration-base factual deltas, CV/tuning:",
      float((zn210-orig_train209)[cvall208].mean()),
      float((ztn210-orig_tune209).mean()))
```
