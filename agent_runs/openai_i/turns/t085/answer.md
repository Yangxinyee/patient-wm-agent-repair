The posterior helper uses the documented context formula, and its final layer can support a reversible low-rank edit. First I’ll replace encoded empty-action vectors with zeros and compare against exposed posterior means. A sharp reduction in observation-only error would confirm that convention; remaining error would point to atom assembly.

```python
import torch
import numpy as np
import pandas as pd

valid77 = np.asarray(valid_indices65, dtype=np.int64)
ctx77 = latent_context(valid77)
target77 = torch.from_numpy(latent_state[valid77, -32:])
u77 = u_all65.clone()
u77[u_counts65 == 0] = 0

reports77 = []
for empty_e77 in ("encoded", "zero"):
    e77 = e_all65.clone()
    if empty_e77 == "zero":
        e77[e_counts65 == 0] = 0
    pred77 = latent_post_mean(ctx77, e77[valid77], u77[valid77])
    err77 = ((pred77 - target77).square().mean(1).sqrt()).numpy()
    report77 = pd.DataFrame({
        "nobs": e_counts65[valid77].numpy().astype(int),
        "nact": u_counts65[valid77].numpy().astype(int),
        "rms": err77
    })
    print("\n=== ZERO ABSENT ACTION; EMPTY OBS =", empty_e77, "===")
    print(report77.groupby(["nobs", "nact"]).rms.agg(["count", "mean", "max"]).to_string())
    reports77.append((empty_e77, float(err77.mean())))

print("\n=== ATOM-COUNT MISMATCH EXAMPLES ===")
for i77 in np.asarray(count_errors65, dtype=int)[:4]:
    print("index", i77, "stay", int(latent_sid[i77]), "hour", float(latent_hour[i77]),
          "encoded counts", float(e_counts65[i77]), float(u_counts65[i77]),
          "expected", int(expected_counts65[i77]),
          "bundle rows", repr(bundle_rows65[i77])[:550])

print("\n=== UNIT MAPPINGS ===")
for name77 in ("measurement_units65", "drug_units65", "drug_rate_units65",
               "unit_ids57", "rate_unit_ids57", "ekind_ids57"):
    print(name77, globals().get(name77))

print("\n=== RECONSTRUCTION MEANS ===", reports77)
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 77; forward 6 — zero-action posterior test\n"
        "- Tested u=0 on absent-action bundles, comparing encoded and zero empty "
        "observation vectors against exposed original posterior means.\n"
        f"- Reconstruction mean RMS by empty-observation convention: {reports77}.\n"
        "- No weights changed.\n"
    )
```