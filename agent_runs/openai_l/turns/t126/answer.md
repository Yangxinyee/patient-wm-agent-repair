Head 2 preserves most dose contrasts, but its factual signal is noisier than the earlier raw-input residual. Head 3 carries essentially no dose contrast. I’ll test whether a correction inserted directly into attention can control the four reading types. Consistent, sensor-specific control with moderate-sized changes would support this route; huge required changes would make it unsuitable.

```python
import numpy as np
import torch

print("=== HEAD-2 EXTRACTOR BY BUNDLE TYPE ===")
b117 = fits116[(2, 0.3)]
fp117 = np.column_stack([features116[:, 2], np.ones(len(valid116))]) @ b117
bundle_drugs117 = np.zeros((len(sx63), 2), dtype=np.int64)
for j117, name117 in enumerate(("insulin", "norepinephrine")):
    np.add.at(bundle_drugs117[:, j117], owners116[drug116 == name117], 1)
for label117, mask117 in (
    ("no drug", bundle_drugs117[valid116].sum(1) == 0),
    ("insulin", bundle_drugs117[valid116, 0] > 0),
    ("norepinephrine", bundle_drugs117[valid116, 1] > 0),
):
    print(label117, "n", int(mask117.sum()),
          "mean", fp117[mask117].mean(0),
          "RMS", np.sqrt(np.mean(fp117[mask117]**2, axis=0)))

print("\n=== ATTENTION OUTPUT SINGULAR VALUES ===")
for hd117 in range(4):
    sv117 = np.linalg.svd(
        w63("core.prelude.0.attn.out.weight")[:, hd117*48:(hd117+1)*48].numpy(),
        compute_uv=False)
    print(hd117, "largest", sv117[:3], "smallest", sv117[-6:])

attention_backup117 = attention63
addition117 = torch.zeros(192)
def attention_inject117(qin, kin, vin, prefix, mask=None):
    result = attention_backup117(qin, kin, vin, prefix, mask)
    if prefix == "core.prelude.0.attn":
        result = result + addition117[None, None, :]
    return result

G117 = np.empty((len(rows93), basis110.shape[1]), dtype=np.float64)
try:
    attention63 = attention_inject117
    zero117 = np.zeros((len(ids93), 32), dtype=np.float32)
    addition117 = torch.zeros(192)
    baseline117 = latent_probe93(zero117)
    for j117 in range(basis110.shape[1]):
        direction117 = np.asarray(basis110[:, j117], dtype=np.float32)
        addition117 = torch.as_tensor(10.0 * direction117)
        plus117 = latent_probe93(zero117)
        addition117 = torch.as_tensor(-10.0 * direction117)
        minus117 = latent_probe93(zero117)
        G117[:, j117] = (plus117 - minus117) / 20.0
finally:
    attention63 = attention_backup117

print("\n=== DIRECT ATTENTION CONTROLLABILITY ===")
print("gradient RMS/max:", np.sqrt(np.mean(G117**2)), np.max(np.abs(G117)))
for name117 in np.unique(sensor_codes110):
    mask117 = sensor_codes110 == name117
    print(name117, "mean-gradient norm", np.linalg.norm(G117[mask117].mean(0)),
          "RMS", np.sqrt(np.mean(G117[mask117]**2)))
desired117 = np.column_stack([
    np.isin(sensor_codes110, ["glucose_finger", "glucose_lab"]).astype(float) * -1.3,
    np.isin(sensor_codes110, ["map_arterial", "map_cuff"]).astype(float) * 3.2,
])
coeff117 = np.linalg.solve(
    G117.T @ G117 + np.eye(G117.shape[1])*1e-8, G117.T @ desired117)
routes117 = np.asarray(basis110) @ coeff117
print("route norms:", np.linalg.norm(routes117, axis=0))
print("linear target RMSE:", np.sqrt(np.mean((G117 @ coeff117-desired117)**2, axis=0)))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 117 / direct-attention controllability\n"
            "Head 2 preserves dose contrasts (I x2≈0.90, NE x2≈1.04 at penalty .3), "
            "but factual residual RMS≈.69/.18. Tested direct attention-output injection "
            "to distinguish weak latent attention from downstream suppression. "
            "Gradient RMS=" + str(float(np.sqrt(np.mean(G117**2)))) +
            "; fitted route norms=" + repr(np.linalg.norm(routes117, axis=0).tolist()) +
            ". No weights edited.\n")
```