The standalone recurrent-input computation agrees closely with the captured one, and the cached inputs align across all bundles. I’ll use it to build dose contrasts from zero through triple rate. Agreement with recorded rates and consistent contrast features would support fitting a residual decoder; mismatches would invalidate that calibration.

```python
print("=== RATE TARGET VALIDATION ===")
print("scales:", list(Uaug168))
print("target range:", np.min(Yd168, axis=0), np.max(Yd168, axis=0))
R193 = np.zeros((len(Uref168), 2), dtype=np.float32)
R193[decode_indices168] = np.asarray(Yd168[:len(decode_indices168)])
print("probe drug labels:", np.unique(drugs182))
for d193, col193 in [("insulin", 0), ("norepinephrine", 1)]:
    mask193 = np.asarray(drugs182) == d193
    if mask193.any():
        print(d193, "rate agreement max error:",
              np.max(np.abs(R193[ids182[mask193], col193] - rates182[mask193])))
print("action row example:", str(act_rows152[0])[:500])
print("target rates at first probes:", R193[ids182[:8]])
print("reference rates:", rates182[:8], drugs182[:8])

ecore193 = {}
zofficial193 = torch.as_tensor(state_cal161)[ix166, -32:]
posterior_base193 = mlp136(torch.cat([
    context, torch.as_tensor(E166), torch.as_tensor(Uref168)], dim=1), "z_post")[:, :32]
for scale193, action193 in Uaug168.items():
    post193 = mlp136(torch.cat([
        context, torch.as_tensor(E166), torch.as_tensor(action193)], dim=1), "z_post")[:, :32]
    z193 = zofficial193 + post193[ix166] - posterior_base193[ix166]
    result193 = torch.empty((len(ix166), 4, 192), dtype=torch.float32)
    for start193 in range(0, len(ix166), 256):
        sl193 = slice(start193, min(start193 + 256, len(ix166)))
        source193 = ix166[sl193]
        tokens193 = torch.cat([
            torch.as_tensor(hf166)[sl193] + W["core.slots"],
            torch.as_tensor(E166)[source193].unsqueeze(1),
            torch.as_tensor(action193)[source193].unsqueeze(1),
            linear136(z193[sl193], "z_in").unsqueeze(1)
        ], dim=1)
        result193[sl193] = norm136(
            block136(tokens193, "core.prelude.0")[:, :4], "core.prelude_norm")
    ecore193[scale193] = result193
print("\n=== PRELUDE DOSE CONTRASTS ===")
base_E193 = ecore193[(1.0, 1.0)]
for scale193, value193 in ecore193.items():
    if scale193 != (1.0, 1.0):
        delta193 = value193 - base_E193
        active193 = np.any(R193[ix166] * (np.asarray(scale193) - 1) != 0, axis=1)
        print(scale193, "active bundles", int(active193.sum()),
              "contrast RMS", float(delta193[active193].square().mean().sqrt()))
print("baseline capture RMS:",
      float((base_E193[loc192] - ecore_base192[shared192]).square().mean().sqrt()))
with open("/work/notes.md", "a") as f193:
    f193.write("\n## Python 193 — recurrent dose contrasts\n"
               "- Built anchored prelude features across zero, half, 1.5x, doubled and tripled doses.\n"
               "- Validating physical-rate targets against the existing drug probes before fitting a decoder.\n"
               "- No weights changed; these features are approximate pathway measurements, not causal labels.\n")
```