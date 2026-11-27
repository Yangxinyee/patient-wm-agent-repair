The correction tracks dose changes with slopes close to one, including stopping and tripling either drug. The feature works; its downstream influence is weak. I’ll install a reversible 100-fold gain probe. A substantial rollout response would confirm that this pathway is usable, but factual fit must still pass.

```python
candidate_small131 = {n: np.asarray(load(n)).copy() for n in edited()}
print("latent direction norms:", float(np.linalg.norm(vg104)), float(np.linalg.norm(vm104)))
for n in NAMES:
    if n.startswith("z_in") or n.startswith("z_head"):
        a = np.asarray(load_original(n))
        print(n, tuple(a.shape), "RMS", float(np.sqrt(np.mean(a*a))))
print("mean posterior correction RMS at small gain:",
      float(np.sqrt(np.mean(delta_base130[:,:32]**2))))

install_repair127(-200.0, 150.0)
repair_gains131 = (-200.0, 150.0)
print("installed gain probe:", repair_gains131)
print("edited tensors:", edited())
print("all edits finite:", all(np.isfinite(np.asarray(load(n))).all() for n in edited()))

with open("/work/notes.md","a") as f:
    f.write("\n## Python 131 — reversible 100-fold latent gain probe\n"
            "- Direct perturbation audit: gated innovation slopes approximately 1 for x0, x.5, x1.5, x2, x3.\n"
            "- Small-gain factual latent correction RMS .00354; dose features are active but rollout influence is weak.\n"
            "- Saved small candidate in candidate_small131; installed gains insulin -200, NE +150.\n"
            "- This is pathway calibration, not an assertion of drug-effect magnitude. Must test rollouts and factual fit.\n")
```
