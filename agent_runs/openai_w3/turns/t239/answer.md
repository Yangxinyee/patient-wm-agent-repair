The insulin carrier is approximately linear across all tested scales, including halving and stopping. Norepinephrine is attenuated, but that does not explain insulin’s asymmetry. I’ll trace the changes through attention and factual compensation; if compensation cancels reductions preferentially, its fit needs stronger contrast constraints.

```python
r214_dirs = np.asarray(directions187, dtype=np.float64)
r214_dual = np.linalg.pinv(r214_dirs).T
r214_w0 = torch.as_tensor(load_original("core.prelude.0.ffn.0.weight"))
r214_b0 = torch.as_tensor(load_original("core.prelude.0.ffn.0.bias"))
r214_w2 = torch.as_tensor(candidate206["core.prelude.0.ffn.2.weight"])
r214_b2 = torch.as_tensor(candidate206["core.prelude.0.ffn.2.bias"])
r214_dw2 = r214_w2-torch.as_tensor(load_original("core.prelude.0.ffn.2.weight"))
r214_db2 = r214_b2-torch.as_tensor(load_original("core.prelude.0.ffn.2.bias"))

def r214_parts(tokens):
    E, after, norm, att = prelude155(tokens, overrides=candidate206)
    hidden = F.gelu(F.linear(
        layer_norm(after, "core.prelude.0.norm3"), r214_w0, r214_b0
    ))
    correction = F.linear(hidden, r214_dw2, r214_db2)
    raw = after + F.linear(hidden, r214_w2, r214_b2)
    return E, after, correction, raw

r214_base = r214_parts(tokens206)
r214_tokens = {}
r214_results = {}
print("=== ATTENTION / COMPENSATION CONTRASTS ===")
print("drug scale | decoded attention compensation total_raw | E_RMS")
for (d, s), u in sorted(scaled_u198.items()):
    tok = tokens206.clone()
    tok[:, 5] = tensor153(u)
    z = posterior153(C154, E154, u, post206)
    tok[:, 6] = F.linear(z, tensor153(zin154), tensor153(zb154))
    parts = r214_parts(tok)
    r214_tokens[(d, s)] = tok
    r214_results[(d, s)] = parts
    mask = (np.asarray(rates187)[:, d] > 1e-8) & np.asarray(valid196, dtype=bool)
    proj = []
    for j in [1, 2, 3]:
        delta = (parts[j]-r214_base[j]).mean(1).detach().cpu().numpy()
        proj.append(float((delta[mask] @ r214_dual)[:, d].mean()))
    decoded = float((r213_signals[(d, s)]-r213_s0)[mask, d].mean())
    erms = float(torch.sqrt(torch.mean(
        (parts[0][torch.as_tensor(mask)]-r214_base[0][torch.as_tensor(mask)])**2
    )))
    print(d, s, [round(x, 6) for x in [decoded]+proj], "E_RMS", round(erms, 7))

print("\n=== LOCAL REPLAY OBJECTS ===")
for k in sorted([
    k for k in list(globals()) if not k.startswith("_")
    and k.endswith(("169", "171", "186"))
    and (hasattr(globals()[k], "shape") or callable(globals()[k]))
]):
    v = globals()[k]
    if callable(v):
        try:
            print(k, "fn", inspect.signature(v))
        except Exception:
            print(k, "fn")
    else:
        print(k, tuple(v.shape))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\nPython run 214; forwards 14. Fixed-context carrier insulin "
        "slopes are 1.004 (stop),1.063 (half),.976 (double),.983 "
        "(triple); NE slopes .70,.825,.687,.663. Insulin carrier alone "
        "does not explain rollout asymmetry. Baseline rollout drift "
        "vs original is glucose +9.08 and MAP -3.28. Testing attention "
        "and FFN compensation contrasts plus local replay. No edits.\n"
    )
```