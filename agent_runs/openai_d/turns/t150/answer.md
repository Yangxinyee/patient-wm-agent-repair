The draw-invariance check failed: the inferred register changed with the latent sample, which it should not do. That means the cached inputs and my reconstructed latent draws are mismatched. No new edit was installed. I’ll identify the matching cached queries, extract the fixed register, and rebuild both sides consistently. Only a draw-invariant register will permit the sensitivity test to proceed.

```python
def r131_numpy(v):
    return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)

print("Prior reconstruction agreement:",
      float(torch.max(torch.abs(r129_pp_o - r124_pp))))
print("--- Input-cache matches ---")
for name in ("xx98", "x106", "X99", "XX99"):
    v = globals()[name]
    if tuple(v.shape) == tuple(r124_x.shape):
        print(name, float(np.max(np.abs(r131_numpy(v) - r124_x.numpy()))))
for key in ("q", "with_code"):
    v = cache98[key]
    print("cache98", key, r126_desc(v))
    if key == "with_code" and tuple(v.shape) == tuple(r124_x.shape):
        print("  max input difference:",
              float(np.max(np.abs(r131_numpy(v) - r124_x.numpy()))))

r131_queries = {
    "reconstructed split prior": r129_q_o,
    "cached qq98": qq98,
    "cache98 q": cache98["q"],
    "fixed cached zz98": r124_norm(
        r127_ctx_o[None] + r124_lin(zz98, "z_head")
        + r124_W["type_emb.weight"][1], "query_norm")
}
r131_errors = {}
for name, q in r131_queries.items():
    q = torch.as_tensor(q, dtype=torch.float32)
    if tuple(q.shape) == tuple(r124_x.shape):
        reg = r124_x - q
        err = float(torch.max(torch.abs(reg - reg[0:1])))
        r131_errors[name] = err
        print(name, "register draw-invariance error:", err)

r131_best = min(r131_errors, key=r131_errors.get)
r131_reg_error = r131_errors[r131_best]
print("Best matching latent/query cache:", r131_best)
if r131_reg_error < 0.001:
    r131_register = (
        r124_x - torch.as_tensor(r131_queries[r131_best], dtype=torch.float32)
    )[0:1].clone()
    r131_x_o = (r129_q_o + r131_register).numpy()
    r131_x_e = (r129_q_e + r131_register).numpy()
    r131_dx = r131_x_e - r131_x_o
    print("Rebuilt paired inputs:", r131_x_o.shape)
    print("Fixed register RMS:", float(torch.sqrt(torch.mean(r131_register**2))))
    print("Head-input probe delta norm mean/max:",
          float(np.linalg.norm(r131_dx, axis=2).mean()),
          float(np.linalg.norm(r131_dx, axis=2).max()))
    for key in ("raw", "register_free"):
        print("Feature cache", key, list(features99[key]))
else:
    print("STOP: no verified fixed register yet.")

print("Vocabulary codes:", VOCAB["codes"])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 131: latent-cache mismatch audit\n"
            f"Best query cache: {r131_best}; register draw-invariance "
            f"error {r131_reg_error:.8g}. No new weights installed. "
            "Rebuilding paired inputs with one consistent prior convention "
            "rather than mixing cached latent conventions.\n")
```