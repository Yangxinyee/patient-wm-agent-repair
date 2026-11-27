That check did not establish anything: it used the wrong inactivity mask and then stopped on a NumPy/Torch conversion error. I’ll correct both and repeat it. The hypothesis predicts that sharper attention will reduce leakage and factual shifts without suppressing active dose signals.

```python
print("=== CORRECTED SHARPENED-ATTENTION CHECK ===")
src242 = np.asarray(src237, dtype=int)
mask242 = np.asarray(inactive237, dtype=bool)[src242]
print("inactive bundles", int(mask242.sum()), "of", len(mask242))
for scale242 in [1.0, 2.0, 4.0]:
    ps242 = torch.softmax(
        torch.log(p241[:, head241, :4, :].clamp_min(1e-38)) * scale242, dim=-1)
    latent242 = ps242[:, :, -1, None] * v241[:, None, -1, response_cols228]
    other242 = torch.einsum(
        "nst,ntc->nsc", ps242[:, :, :-1], v241[:, :-1, response_cols228])
    sig242 = array241(latent242 + other242)
    print("scale", scale242,
          "inactive RMS", np.sqrt(np.mean(sig242[mask242] ** 2)),
          "inactive max", np.max(np.abs(sig242[mask242])),
          "active RMS", np.sqrt(np.mean(sig242[~mask242] ** 2)))

print("\n=== CORRECTED LOCAL FACTUAL RESPONSE CHECK ===")
base242 = array241(base_means228[0])[idx241]
base_h242 = array241(base_h214)[idx241]
trial242 = {}
for scale242 in [1.0, 2.0, 4.0]:
    cand242 = response_candidate231("mean", 1.0)
    q242 = np.array(cand242["core.prelude.0.attn.q.weight"], copy=True)
    q242[head241 * 48:(head241 + 1) * 48] *= scale242
    cand242["core.prelude.0.attn.q.weight"] = q242
    h242, o242 = replay_route228(cand242, indices=idx241)
    means242, logits242 = readout214(h242, o242, indices=idx241)
    delta242 = array241(means242) - base242
    trial242[scale242] = (
        cand242, array241(means242), array241(logits242))
    print("scale", scale242,
          "mean shifts", np.mean(delta242, axis=0),
          "RMS shifts", np.sqrt(np.mean(delta242 ** 2, axis=0)),
          "slot RMS", np.sqrt(np.mean((array241(h242) - base_h242) ** 2)))

print("\n=== DOSE CONTRAST SHAPES ===")
for key242, val242 in contrast214.items():
    aa242 = np.asarray(val242)
    print(key242, aa242.shape, str(aa242.reshape(-1)[:5])[:140])

with open("/work/notes.md", "a") as f242:
    f242.write("\n## Python 241–242 — repair selector diagnostic\n"
               "- Run241 used an incorrect inactivity mask and stopped on mixed NumPy/Torch subtraction.\n"
               "- Repeating with the retained all-bundle mask and explicit array conversion.\n"
               "- Neutral209 remains installed; no response edit active.\n")
```
