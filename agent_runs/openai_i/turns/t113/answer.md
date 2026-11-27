The output confirms the bundle structure, but it still does not validate the reconstructed encoder features. I’ll first estimate glucose and MAP directions using only the exposed original posterior states and recorded readings, adjusting for prior context. Agreement across held-out stays would support using these directions for an edit; unstable directions would rule that out.

```python
from sklearn.linear_model import Ridge
import torch.nn.functional as F

torch.set_num_threads(8)
S104 = np.asarray(state_x, dtype=np.float32)
ids104 = np.asarray(state_ids)
valid104 = np.flatnonzero(np.r_[False, ids104[1:] == ids104[:-1]])
ctx104 = latent_context(valid104).detach().numpy()
z104 = S104[valid104, 960:992]

sensor_codes104 = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
readings104 = np.full((len(bundle_rows65), 4), np.nan, dtype=np.float32)
for i104, atoms104 in enumerate(bundle_rows65):
    for kind104, row104 in atoms104:
        if kind104 == 0 and row104["event"] in sensor_codes104:
            readings104[i104, sensor_codes104.index(row104["event"])] = float(row104["value"])

r104 = readings104[valid104]
flags104 = np.isfinite(r104)
hasg104 = flags104[:, :2].any(1)
hasm104 = flags104[:, 2:].any(1)
g104 = np.nansum(r104[:, :2], axis=1) / np.maximum(flags104[:, :2].sum(1), 1)
m104 = np.nansum(r104[:, 2:], axis=1) / np.maximum(flags104[:, 2:].sum(1), 1)
gcentre104 = np.median(g104[hasg104])
mcentre104 = np.median(m104[hasm104])
levels104 = np.column_stack([
    np.where(hasg104, g104-gcentre104, 0),
    np.where(hasm104, m104-mcentre104, 0),
    flags104.astype(float)
])
Xdir104 = np.column_stack([ctx104, levels104])
readmask104 = hasg104 | hasm104
trainmask104 = readmask104 & ((ids104[valid104] % 5) != 0)
testmask104 = readmask104 & ((ids104[valid104] % 5) == 0)

direction_fit104 = Ridge(alpha=2.0).fit(Xdir104[trainmask104], z104[trainmask104])
direction_all104 = Ridge(alpha=2.0).fit(Xdir104[readmask104], z104[readmask104])
vg104 = direction_all104.coef_[:, 192].astype(np.float32)
vm104 = direction_all104.coef_[:, 193].astype(np.float32)

def cosine104(a, b):
    return float(a @ b / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))

print("READING BUNDLES: train, held out", trainmask104.sum(), testmask104.sum())
print("HELD-OUT LATENT R2", direction_fit104.score(Xdir104[testmask104], z104[testmask104]))
print("GLUCOSE: direction norm, split agreement",
      np.linalg.norm(vg104), cosine104(vg104, direction_fit104.coef_[:, 192]))
print("MAP: direction norm, split agreement",
      np.linalg.norm(vm104), cosine104(vm104, direction_fit104.coef_[:, 193]))
print("GLUCOSE/MAP DIRECTION COSINE", cosine104(vg104, vm104))
print("EDITED", edited())

with Path("/work/notes.md").open("a") as f104:
    f104.write("\n## Python 104 — observation-linked latent directions\n"
               "- Used exposed original posterior means, recorded current readings, and pre-bundle context.\n"
               "- Checked latent-direction stability across disjoint stays.\n"
               "- These directions are an editing pathway, not evidence of drug effects. No weights changed.\n")
```
