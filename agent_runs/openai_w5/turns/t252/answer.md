The helper appears to use the previously selected register layout, but its remaining calculations still need checking. No weights have changed. I’ll compare original and edited head reconstructions, including latent averaging, and inspect the saved calibration coefficients. This will distinguish a reconstruction error from an unsuitable causal correction.

```python
print("=== FORWARD FILE INDEX ===")
for p in sorted(Path("/work").glob("fwd_*"), key=lambda p: int(p.name.split("_")[-1])):
    print(p.name, ",".join(q.name for q in p.iterdir()))
print("state IDs", state_ids[:5], state_ids[-5:])

print("\n=== HELPER'S FINAL OPERATIONS ===")
ins223 = list(dis.get_instructions(make_head_inputs212))
for insn in ins223:
    if insn.offset >= 580:
        print(insn.opname, str(insn.argval)[:75])

print("\n=== CALIBRATION OBJECT SUMMARIES ===")
def brief224(ob, depth=0):
    if isinstance(ob, dict):
        return {str(k): brief224(v, depth+1) for k, v in ob.items()}
    if isinstance(ob, (torch.Tensor, np.ndarray)):
        ar = ob.detach().cpu().numpy() if isinstance(ob, torch.Tensor) else ob
        return {"shape": ar.shape, "mean": float(np.nanmean(ar)),
                "std": float(np.nanstd(ar))} if ar.size else {"shape": ar.shape}
    if isinstance(ob, (list, tuple)):
        return [brief224(x, depth+1) for x in ob] if len(ob) <= 12 else {"length": len(ob)}
    return str(ob)[:160]
for nm in ["head_audit", "history_audit", "encoder_audit"]:
    print(nm, str(brief224(globals()[nm]))[:1700])
```
