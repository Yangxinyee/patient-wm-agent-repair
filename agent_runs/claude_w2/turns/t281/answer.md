The kernel still holds a lot from runs 208 to 265:
- `designs` and `SOL`: candidate solutions for the channel design;
- `Dx`, `dy_`, `r2d`: state differences across dose arms, with an R² value;
- `res_w`, `res_b`, `res_n`: rollout results;
- `Cs`, `Cq`: probably coefficient sets.

chan.py is an elaborate edit that writes the dose into the encoder, the prelude attention and the core. Its comments show it was still at the design stage.

The cheaper alternative is to read dose information the model already holds and add it in the value head. Whether that works depends on one number: how well the existing state tracks dose changes under intervention. `r2d`, the R² of decoding dose differences from state differences, likely answers this.

I'll print these stored results compactly, then decide:
- **High R² (above about 0.8):** go head-only.
- **Low R²:** the core needs a dedicated memory channel.

```python
def brief(v):
    if hasattr(v, 'shape') and getattr(v, 'ndim', 1) == 0: return f"{float(v):.4g}"
    if hasattr(v, 'shape'):
        a = np.asarray(v, dtype=float) if not torch.is_tensor(v) else v.double().numpy()
        return f"arr{tuple(a.shape)}" + (f"={np.round(a.ravel()[:6],3).tolist()}" if a.size <= 12 else f" mean{a.mean():.3g}")
    if isinstance(v, (int, float, np.floating)): return f"{float(v):.4g}"
    if isinstance(v, dict): return "dict{" + ",".join(list(map(str, v.keys()))[:8]) + "}"
    if isinstance(v, (list, tuple)): return f"{type(v).__name__}[{len(v)}]"
    return type(v).__name__
print("r2d", r2d, "res_on", res_on, "mn_on", mn_on, "mx_off", mx_off, "sbar", sbar)
for nm in ['Cs', 'Cq', 'E1', 'designs', 'SOL', 'res_w', 'res_b', 'res_n', 'Dw']:
    d = globals()[nm]
    print(f"== {nm}:", "; ".join(f"{k}: {brief(v)}" for k, v in list(d.items())[:10])[:700])
print("armsW", armsW, "\narms_i", arms_i, "\narms_n", arms_n)
```