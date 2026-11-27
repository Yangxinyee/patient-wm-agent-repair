The helpers expose all inputs needed to test the empty-action explanation directly. I’ll recompute posterior states with zero action vectors and compare them with the saved states. If observation-only errors remain, I’ll inspect bundle construction and time context rather than keep changing activations.

```python
import dis
import pandas as pd

rows91 = np.asarray(posterior_rows_repair, dtype=int)
truth91 = torch.as_tensor(refx[rows91, 960:992], dtype=torch.float32)
mask91 = np.asarray(has_action85, dtype=bool)
print("=== EMPTY ACTION TEST ===")
print("Existing action-vector RMS, no-action bundles:",
      float(posterior_u_repair[~mask91].square().mean().sqrt()))
for label91, e91 in [("original_e", posterior_e_repair),
                     ("v2_e", posterior_e_v2)]:
    for mode91 in ["existing_u", "zero_when_absent"]:
        u91 = posterior_u_repair.clone()
        if mode91 == "zero_when_absent":
            u91[~mask91] = 0
        z91 = rmlp(torch.cat([posterior_ctx_repair, e91, u91], dim=1), "z_post")[:, :32]
        err91 = (z91 - truth91).square().mean(1).sqrt().numpy()
        print(label91, mode91,
              "no-action mean/max", float(err91[~mask91].mean()), float(err91[~mask91].max()),
              "action mean/max", float(err91[mask91].mean()), float(err91[mask91].max()))

print("\n=== CONTEXT FUNCTION ===")
dis.dis(previous_context_repair)
print("\n=== SET ENCODER FUNCTION ===")
dis.dis(encode_atom_set)
print("\n=== SAMPLE STATE METADATA ===")
print(state_meta.head(12).to_string(index=False))
print("\n=== SAMPLE OBSERVATION SETS ===")
for i91 in np.flatnonzero(~mask91)[:3]:
    print("index", int(i91), "state row", int(rows91[i91]),
          "observations", obs_sets_repair[i91],
          "actions", action_records_repair[i91])
```
