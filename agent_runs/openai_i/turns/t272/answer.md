Doubling selector strength reduced inactive signal RMS about fortyfold, while leaving active signals and local response predictions unchanged. That supports the leakage explanation, but not yet factual safety. I’ll install this sharper candidate provisionally; the GPU factual test must stay within +0.005 NLL.

```python
candidate243 = {
    k: np.asarray(v, dtype=np.float32).copy()
    for k, v in trial242[2.0][0].items()
}
for name243 in list(edited()):
    drop_edit(name243)
for name243, value243 in candidate243.items():
    save_edit(name243, value243)
np.savez("/work/sharp243.npz", **candidate243)

print("Installed:", edited())
print("Response-column norms:",
      np.linalg.norm(candidate243["core.prelude.0.attn.out.weight"][:, response_cols228],
                     axis=0))
print("Selector row norm:",
      np.linalg.norm(candidate243["core.prelude.0.attn.q.weight"][head241 * 48]))

with open("/work/notes.md", "a") as f243:
    f243.write("\n## Python 243 — install sharper response candidate provisionally\n"
               "- Doubling the selector reduced inactive attention RMS from 9.77e-5 to 2.24e-6.\n"
               "- Active transport and local factual mean shifts were unchanged.\n"
               "- Installed sharp243: response233 with doubled reserved-head selector.\n"
               "- Not certified: requires GPU factual dNLL <= .005 before rollout consideration.\n")
```
