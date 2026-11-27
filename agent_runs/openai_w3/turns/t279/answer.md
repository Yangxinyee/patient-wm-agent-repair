The earlier validated checkpoint used a much smaller FFN compensator and produced better-balanced effects than the later refit. I’ll use that checkpoint for a reversible test, scaling the actual attention payload to 25% for insulin and 150% for norepinephrine. Improved insulin symmetry would support saturation; proportional shrinkage of both insulin effects would argue against it.

```python
checkpoint249 = {n: np.asarray(load(n)).copy() for n in edited()}
candidate249 = {n: np.asarray(v).copy() for n, v in checkpoint193.items()}
out_name249 = "core.prelude.0.attn.out.weight"
zero_out249 = np.asarray(zero169[out_name249], dtype=np.float64)
payload249 = np.asarray(checkpoint193[out_name249], dtype=np.float64) - zero_out249
carrier_rows249 = np.linalg.lstsq(dd193, payload249, rcond=1e-10)[0]
payload_residual249 = payload249 - dd193 @ carrier_rows249
relative_residual249 = np.linalg.norm(payload_residual249) / np.linalg.norm(payload249)
print("Payload rank-two reconstruction relative error:", relative_residual249)
print("Carrier row norms:", np.linalg.norm(carrier_rows249, axis=1))
gains249 = np.array([0.25, 1.5])
assert relative_residual249 < 0.01, "Payload decomposition failed; no candidate installed."
candidate249[out_name249] = (
    zero_out249 + (dd193 * gains249[None, :]) @ carrier_rows249 + payload_residual249
).astype(np.float32)
for n249, a249 in candidate249.items():
    assert tuple(a249.shape) == tuple(SHAPES[n249]) and np.isfinite(a249).all()
for n249 in list(edited()):
    if n249 not in candidate249:
        drop_edit(n249)
for n249, a249 in candidate249.items():
    save_edit(n249, torch.from_numpy(a249.copy()))
print("Installed candidate249; actual attention payload gains:", gains249)
print("Attention payload norm:", np.linalg.norm(candidate249[out_name249] - zero_out249))
print("FFN compensation weight norm:", np.linalg.norm(candidate249["core.prelude.0.ffn.2.weight"]))
with open("/work/notes.md", "a") as f249:
    f249.write("\nPython run 249; forwards 18. Installed reversible candidate249 based on validated checkpoint193, "
               "scaling actual rank-two attention payload to insulin .25 and NE 1.5, retaining checkpoint193's small FFN compensator. "
               f"Payload decomposition relative error {relative_residual249:.3g}. "
               "Checkpoint249 retains validated checkpoint222. Testing saturation at the actual payload, not another FFN correction.\n")
```