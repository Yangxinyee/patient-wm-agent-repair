The sensor gate is accurate to about 0.001 on the cached factual features, so it looks promising for separating glucose and MAP corrections. I’ll now check whether norepinephrine’s large decoder error comes mainly from irrelevant glucose features. Small error on MAP rows would support gating; large error there would mean the dose decoder itself needs improvement.
```python
print("=== RECORD-EFFECT NOTES ===")
print(Path("/work/notes.md").read_text()[:6000])

print("\n=== DOSE DIRECTION BY SENSOR ===")
for drug in ["insulin", "norepinephrine"]:
    for lam in [.03, .1, .3, 1.0]:
        w = directions193[(drug, lam)]
        p = XF192 @ w
        summary = {}
        for ev in np.unique(events179):
            a = p[:, (events179 == ev) & ~fitmask179]
            summary[ev] = (
                round(float(np.mean(a)), 3),
                round(float(np.std(a)), 3),
                round(float(np.sqrt(np.mean(a*a))), 3))
        print(drug, lam, "held-out (mean, SD, RMS)", summary)

print("\n=== FEATURE / TARGET ALIGNMENT ===")
for k in [("insulin", 1.0), ("insulin", 2.0),
          ("norepinephrine", 1.0), ("norepinephrine", 2.0)]:
    c = cf_features81[k]
    print(k, {s: getattr(v, "shape", None) for s,v in c.items()})
    if k in targets179:
        describe196("target " + str(k), targets179[k])
print("preactivation reconstruction error",
      float(np.max(np.abs(arr192(torch.nn.functional.gelu(
          torch.tensor(I192 @ W0192.T + b0192, dtype=torch.float32))) - H192))))
for n in ["nll179", "target_logits83"]:
    ins = list(dis.get_instructions(globals()[n]))
    print(n, "return instructions",
          [(i.opname, i.argrepr) for i in ins[-16:]])
for n in ["Lorig179", "L179"]:
    z = nll179(globals()[n])
    print(n, "nll return", type(z).__name__, getattr(z, "shape", None),
          "mean", float(np.mean(arr192(z))))
print("quiet hidden rows", quiet192.tolist())
```
