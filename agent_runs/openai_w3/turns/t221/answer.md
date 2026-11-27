The prelude preserves insulin contrasts in both directions, but the full-latent decoder could also respond to unrelated state changes. Two reserved carriers are isolated from the original latent features. I’ll test a decoder restricted to those carriers; success would mean useful dose contrasts and essentially zero decoded dose on bundles without drugs.

```python
gamma199 = as_numpy(load_original("core.prelude.0.norm1.weight")).astype(float)
beta199 = as_numpy(load_original("core.prelude.0.norm1.bias")).astype(float)
const199 = (1.0 / gamma199)
const199 /= float(beta199 @ const199)

restricted0_199 = carrier154 / gamma199[:, None] * (
    sigma_ref154 / carrier_scale154)
restricted0_199 -= const199[:, None] * (beta199 @ restricted0_199)[None, :]

print("=== RESTRICTED CARRIER DECODER ===")
print("carrier sums:", carrier154.sum(axis=0))
print("constant-channel value:", float(beta199 @ const199))
print("decoder norms:", np.linalg.norm(restricted0_199, axis=0))

restricted199 = restricted0_199.copy()
for d in [0, 1]:
    num, den = 0.0, 0.0
    for s in [0.0, 0.5, 1.5, 2.0, 3.0]:
        dx, dy = pairs187[(d, s)]
        use = np.asarray(train187, bool) & (np.abs(dy[:, d]) > 1e-8)
        xx = (dx @ restricted0_199)[use, d]
        yy = dy[use, d]
        weight = 1.0 / max(float(np.mean(yy**2)), 1e-8)
        num += weight * float(xx @ yy)
        den += weight * float(xx @ xx)
    restricted199[:, d] *= num / den
    print("fitted gain", d, num / den)

for d in [0, 1]:
    for s in [0.0, 0.5, 1.5, 2.0, 3.0]:
        dx, dy = pairs187[(d, s)]
        use = np.asarray(valid187, bool) & (np.abs(dy[:, d]) > 1e-8)
        p = (dx @ restricted199)[use, d]
        y = dy[use, d]
        print(d, s, "slope", round(float(p @ y / (y @ y)), 4),
              "means", np.round([p.mean(), y.mean()], 5),
              "RMSE", round(float(np.sqrt(np.mean((p-y)**2))), 5))

xlatent199 = np.asarray(latent187(U154))
nodrug199 = np.linalg.norm(np.asarray(U154), axis=1) < 1e-8
print("no-drug n:", int(nodrug199.sum()))
print("no-drug restricted/full RMS:",
      np.sqrt(np.mean((xlatent199[nodrug199] @ restricted199)**2, axis=0)),
      np.sqrt(np.mean(((xlatent199[nodrug199]-mu187) @ decoder192)**2, axis=0)))

print("\n=== RESERVED ATTENTION LAYOUT ===")
dkv199 = np.asarray(checkpoint193["core.prelude.0.attn.kv.weight"]) - np.asarray(
    base192["core.prelude.0.attn.kv.weight"])
do199 = np.asarray(checkpoint193["core.prelude.0.attn.out.weight"]) - np.asarray(
    base192["core.prelude.0.attn.out.weight"])
rows199 = np.where(np.linalg.norm(dkv199, axis=1) > 1e-7)[0]
cols199 = np.where(np.linalg.norm(do199, axis=0) > 1e-7)[0]
print("changed KV rows:", rows199,
      "norms:", np.linalg.norm(dkv199[rows199], axis=1))
print("changed output columns:", cols199,
      "norms:", np.linalg.norm(do199[:, cols199], axis=0))
for row in rows199:
    print("KV row", row, "distance to original decoder columns:",
          np.linalg.norm(dkv199[row, :, None] - centred192, axis=0))
print("reserved head latent attention:",
      arr183(p0_198[3])[:, :, :, -1].mean(axis=(0, 2)))
print("base KV row norms:", np.linalg.norm(
    np.asarray(base192["core.prelude.0.attn.kv.weight"])[rows199], axis=1))
print("base output column norms:", np.linalg.norm(
    np.asarray(base192["core.prelude.0.attn.out.weight"])[:, cols199], axis=0))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 199; forwards 12. Actual prelude insulin contrasts remain bidirectional. Testing an exact two-carrier-only decoder that cannot respond to original latent features, with explicit zero on no-drug bundles, before a reversible rollout pilot. No weights changed.\n")
```