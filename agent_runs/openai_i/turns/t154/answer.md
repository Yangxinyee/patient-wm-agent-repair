The bin and encoder-activation tests did not remove the discrepancy. Earlier notes say exact reconstruction was achieved, so I’ll recover those conventions and separately test the time-context calculation. A common near-zero error across sensors would validate the reconstruction; sensor-specific compensations would not.

```python
print("=== PREVIOUS EXACT-RECONSTRUCTION NOTES ===")
for sec139 in sections135:
    match139 = re.search(r"Python (\d+)", sec139[:100])
    if match139 and 54 <= int(match139.group(1)) <= 61:
        print(sec139[:900])

ix139 = sum([ix for ix in probe_ix138.values()], [])
events139 = [bundles[i]["starts"][0]["event"] for i in ix139]
unit139 = {"glucose_finger": 1, "glucose_lab": 5,
           "map_arterial": 2, "map_cuff": 2}
atoms139 = torch.stack([
    atom138(bundles[i]["starts"][0], unit139[event])
    for i, event in zip(ix139, events139)])
e139 = norm136(
    mlp136(mlp136(atoms139, "encode.mlp1")
           + linear136(torch.tensor([np.log(2)], dtype=torch.float32), "encode.size"),
           "encode.mlp2"),
    "encode.norm")
u139 = pool137([], "encode_action").expand(len(ix139), -1)
target139 = torch.from_numpy(S[ix139, 960:])
hprev139 = torch.from_numpy(S[previous[ix139], 192:960].reshape(-1, 4, 192))
outprev139 = torch.from_numpy(S[previous[ix139], :192])
dt139 = torch.tensor(
    [(round(hours[i] * 60) - round(hours[previous[i]] * 60)) * 60
     for i in ix139], dtype=torch.float32)
flow139 = hprev139 * torch.exp(
    -dt139[:, None, None] * torch.exp(W["core.flow.a"])[None, None, :])
base139 = outprev139 + norm136(flow139.mean(1), "core.out_norm")

context_results139 = []
for activation139 in ["gelu", "silu", "relu"]:
    for scale139 in [0.1, 1.0]:
        for divisor139 in [1.0, 60.0, 3600.0]:
            for log_mode139 in ["log", "log1p"]:
                t139 = dt139 / divisor139
                argument139 = (torch.log(t139.clamp_min(1e-8))
                               if log_mode139 == "log" else torch.log1p(t139))
                ctx139 = base139 + mlp138(
                    argument139[:, None] * scale139, "gap", activation139)
                pred139 = mlp136(torch.cat([ctx139, e139, u139], -1),
                                 "z_post")[:, :32]
                rms139 = float(((pred139 - target139) ** 2).mean().sqrt())
                context_results139.append(
                    (rms139, activation139, scale139, divisor139, log_mode139))
context_results139.sort()
print("\n=== TIME-CONTEXT CONVENTIONS ===")
print(context_results139[:10])

best139 = context_results139[0]
_, act139, scale139, divisor139, logmode139 = best139
t139 = dt139 / divisor139
arg139 = torch.log(t139.clamp_min(1e-8)) if logmode139 == "log" else torch.log1p(t139)
bestctx139 = base139 + mlp138(arg139[:, None] * scale139, "gap", act139)
pred139 = mlp136(torch.cat([bestctx139, e139, u139], -1), "z_post")[:, :32]
res139 = (pred139 - target139).numpy()
print("residual RMS:", float(np.sqrt(np.mean(res139 ** 2))))
print("RMS after removing coordinate means:",
      float(np.sqrt(np.mean((res139 - res139.mean(0)) ** 2))))
print("encoder MLP shapes:",
      [(n, tuple(W[n].shape)) for n in NAMES
       if n.startswith("encode.") or n.startswith("encode_action.")])
```