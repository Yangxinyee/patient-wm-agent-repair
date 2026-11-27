The cached tokens cover stopping, halving, doubling, and tripling either drug, and the calibration targets include all four sensors. Those are useful constraints, but I still need to verify that the caches match the installed edit. I’ll check that alignment and the prelude return values next; any mismatch would invalidate a regression built from them.

```python
import torch

tokens259 = torch.as_tensor(rawtokens146).clone()
tokens259[:, 5] = u_factual255
tokens259[:, 6] = zin_factual255
parts259 = parts254(tokens259)
oparts259 = parts254(torch.as_tensor(rawtokens146), original=True)

print("=== PARTS RETURN ALIGNMENT ===")
for j, p in enumerate(parts259):
    print("part", j, "shape", tuple(p.shape),
          "RMS", float(torch.sqrt(torch.mean(p.float()**2))))
    for label, cached in [("after254", after254), ("hidden254", hidden254),
                          ("raw254", raw254), ("E254", E254)]:
        if p.shape == cached.shape:
            print("  vs", label, "RMS",
                  float(torch.sqrt(torch.mean((p-cached)**2))))
print("original parts:", [tuple(p.shape) for p in oparts259])

print("\n=== CACHED TOKEN ALIGNMENT ===")
print("original action vs factual RMS:",
      float(torch.sqrt(torch.mean((torch.as_tensor(rawtokens146)[:,5]-u_factual255)**2))))
print("r214 keys:", list(r214_tokens))
for key in [(0,0.5), (0,2.0), (1,0.5), (1,2.0)]:
    t = torch.as_tensor(r214_tokens[key])
    print(key, "unchanged first five token RMS",
          float(torch.sqrt(torch.mean((t[:,:5]-tokens259[:,:5])**2))),
          "action difference RMS", float(torch.sqrt(torch.mean((t[:,5]-tokens259[:,5])**2))))
    if key == (0,2.0):
        q = latent206(t[:,5])
        print("latent206 result:",
              [tuple(v.shape) for v in q] if isinstance(q,(tuple,list)) else tuple(q.shape))

print("\n=== FIT CACHE AND HEAD SHAPES ===")
print("X220 mean/std/end:", X220[:,0].mean(), X220[:,0].std(),
      np.min(X220[:,-1]), np.max(X220[:,-1]))
print("repair physical dose ranges:", repair_doses243.min(0), repair_doses243.max(0))
print("train/holdout slots:", int(tr220.sum()), int(va220.sum()))
for n in NAMES:
    if any(s in n for s in ["attr_heads.value", "last_numeric", "z_head", "core.out_norm"]):
        print(n, SHAPES[n])
print("checkpoint222 current match:", all(torch.equal(
    torch.as_tensor(load(n)), torch.as_tensor(v)) for n,v in checkpoint222.items()))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 259; forwards 20. Verifying factual and multi-scale changed-dose cache alignment and prelude return ordering before jointly constraining the total raw-state response. No weights changed.\n")
```