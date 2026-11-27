The cached inputs agree exactly on the reading-register contribution, and the baseline states are available. I’ll verify the remaining reconstruction, then temporarily shift norepinephrine’s numeric-dose embedding direction. This is a probe, not a treatment-effect estimate or final edit: it would be useful only if the resulting state direction supports a correction that later passes the actual dose rollout.

```python
r124_W = {n: load_original(n).clone() for n in NAMES}
def r124_lin(x, p):
    return F.linear(x, r124_W[p + ".weight"], r124_W.get(p + ".bias"))
def r124_mlp(x, p):
    return r124_lin(F.gelu(r124_lin(x, p + ".0")), p + ".2")
def r124_norm(x, p):
    return F.layer_norm(x, (192,), r124_W[p + ".weight"], r124_W[p + ".bias"])
def r124_context(s, meta):
    idx = meta.state_idx.to_numpy(dtype=np.int64)
    a = torch.as_tensor(s["x"][idx], dtype=torch.float32)
    dt = torch.as_tensor(meta.dt.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    h = a[:, 192:960].reshape(-1, 4, 192)
    hf = h * torch.exp(-dt[:, :, None] * torch.exp(r124_W["core.flow.a"])[None, None, :])
    return a[:, :192] + r124_norm(hf.mean(1), "core.out_norm") + r124_mlp(torch.log(dt) / 10, "gap")

r124_ctx = r124_context(r123_states, meta96)
r124_ctx_error = (r124_ctx - ctx98).abs().max().item()
r124_x = r124_norm(
    r124_ctx[None] + r124_lin(zz98, "z_head") + r124_W["type_emb.weight"][1],
    "query_norm") + fixed96[None]
r124_x_error = (r124_x - xx98).abs().max().item()
r124_pp = r124_mlp(r124_ctx, "z_prior")
print("context max error", r124_ctx_error)
print("head input max error", r124_x_error)
print("prior max error", (r124_pp - pp98).abs().max().item())
print("prior mean max error", (r124_pp[:, :32] - pm98).abs().max().item())
print("log-sd ranges", r124_pp[:, 32:].min().item(),
      r124_pp[:, 32:].max().item(), ls98.min().item(), ls98.max().item())
assert r124_ctx_error < 2e-4 and r124_x_error < 2e-4

r124_probe_delta_rate = 0.10
r124_probe_code = r124_W["embed.code.weight"].clone()
r124_probe_code[9] += (r124_probe_delta_rate / 0.59) * (
    r124_W["embed.numeric.weight"][:, 2] + r124_W["embed.numeric.weight"][:, 4])
save_edit("embed.code.weight", r124_probe_code)
print("temporary NE code-vector shift norm",
      (r124_probe_code[9] - r124_W["embed.code.weight"][9]).norm().item())
print("edits", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 124: controlled NE representation probe\n"
            "Verified exact head-input reconstruction against cached original states. "
            "Temporarily shifted only NE embed.code by the continuous numeric direction "
            "corresponding to +0.10 rate. This does not change bins and may also affect END "
            "atoms, so it is a diagnostic direction, not the actual dose_edit intervention. "
            "Must restore embed.code immediately after paired states are obtained.\n")
```
