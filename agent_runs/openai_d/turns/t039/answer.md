The register layout matches the cached reconstruction, but the state-derived view and fixed inputs do not. That points to an implementation mismatch rather than evidence that dose information is absent. I’ll locate the exact state packing and timing conventions; matching the cached tensors will confirm the reconstruction is usable.

```python
print("=== STATE PACKING CHECK ===")
print("Packed state overlap error:",
      float((wr_x[wr_overlap] - rh_x).abs().max()))
out_errors = []
for offset in [0, 192, 384, 576, 768]:
    err = float((wr_x[wr_overlap, offset:offset+192]
                 - rh_out).square().mean().sqrt())
    out_errors.append((err, offset))
print("out offsets:", out_errors)
h_errors = []
for offset in [0, 192]:
    err = float((wr_x[wr_overlap, offset:offset+768].reshape(-1, 4, 192)
                 - rh_h).square().mean().sqrt())
    h_errors.append((err, offset))
print("h offsets:", h_errors)

print("\n=== CODE / BIN CHECK ===")
print("Code mismatches:", int((wr_codes[wr_overlap] != rh_codes).sum()))
print("Last-bin mismatches:", int((wr_lb[wr_overlap] != rh_lb).sum()))
mismatch = (wr_lb[wr_overlap] != rh_lb).numpy()
if mismatch.any():
    print(rh_meta.loc[mismatch, ["event", "last_z", "last_bin"]]
          .assign(cached_bin=rh_lb[mismatch].numpy()).head(12).to_string(index=False))
print("rh time inputs:")
for key in ["hours_div10", "seconds_div10", "hours", "seconds"]:
    print(key, rh_times[key][:3].flatten().tolist())

wr_out_offset = min(out_errors)[1]
wr_h_offset = min(h_errors)[1]
wr_out = wr_x[:, wr_out_offset:wr_out_offset+192]
wr_h = wr_x[:, wr_h_offset:wr_h_offset+768].reshape(-1, 4, 192)
wr_flow = wr_h * torch.exp(
    -wr_dt[:, None, :] * torch.exp(W0["core.flow.a"])[None, None, :])
wr_flowbase = wr_out + rc_norm(wr_flow.mean(1), "core.out_norm")

wr_best_prior = sorted(rh_prior_trials, key=lambda x: x[0])[0]
print("\n=== PRIOR AND VIEW CONVENTION CHECK ===")
wr_times = {
    "hours_div10": torch.log1p(wr_dt / 3600) / 10,
    "seconds_div10": torch.log1p(wr_dt) / 10,
    "hours": torch.log1p(wr_dt / 3600),
    "seconds": torch.log1p(wr_dt),
}
wr_gaps = {key: rc_mlp(val, "gap", "gelu")
           for key, val in wr_times.items()}
for key, gapj in wr_gaps.items():
    priorj = rc_mlp(wr_flowbase + gapj, "z_prior", "gelu")
    print("prior", key, "RMS mismatch to cached best:",
          float((priorj[wr_overlap] - wr_best_prior[3])
                .square().mean().sqrt()))

wr_prior_gap = min(wr_gaps, key=lambda key: float(
    (rc_mlp(wr_flowbase + wr_gaps[key], "z_prior", "gelu")[wr_overlap]
     - wr_best_prior[3]).square().mean()))
wr_prior = rc_mlp(wr_flowbase + wr_gaps[wr_prior_gap], "z_prior", "gelu")
wr_mu = wr_prior[:, :32]
for key, gapj in wr_gaps.items():
    qj = rc_norm(wr_flowbase + gapj + rc_lin(wr_mu, "z_head")
                 + W0["type_emb.weight"][1], "query_norm")
    print("view", key, "RMS query mismatch to cached best:",
          float((qj[wr_overlap] - wr_best_prior[4])
                .square().mean().sqrt()))

print("Detected packing:", {"out": wr_out_offset, "h": wr_h_offset},
      "prior gap:", wr_prior_gap)
print("Current edits:", edited())
```
