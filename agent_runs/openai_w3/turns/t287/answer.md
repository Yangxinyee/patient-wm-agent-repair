The fixed features match the installed model essentially exactly, so they can support a reliable regression. I’ll calibrate drug-response directions around the original factual states and verify the dose encoder. Opposite-signed local responses with limited saturation would justify fitting total factual and changed-dose states together.

```python
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
torch.set_num_threads(8)

u_factual255 = torch.as_tensor(action_vectors87((1.0, 1.0)), dtype=torch.float32)
z_factual255 = posterior153(C154, E154, u_factual255, post206)
zin_factual255 = F.linear(torch.as_tensor(z_factual255, dtype=torch.float32),
                         load("z_in.weight"), load("z_in.bias"))
print("Factual action / latent token RMS:",
      float((u_factual255 - tokens206[:, 5]).square().mean().sqrt()),
      float((zin_factual255 - tokens206[:, 6]).square().mean().sqrt()))

dist255 = torch.cdist(
    tokens_orig169[:, :4].reshape(len(tokens_orig169), -1).float(),
    tokens206[:, :4].reshape(len(tokens206), -1).float()
)
indices_cal255 = dist255.argmin(dim=1)
print("Calibration slot-match RMS:",
      float((tokens_orig169[:, :4] -
             tokens206[indices_cal255, :4]).square().mean().sqrt()))

saved_raw255 = r233_raw
r233_raw = orig_raw220[indices_cal255].clone()
try:
    zero255 = np.asarray(response_raw234(np.zeros((1, 192), np.float32)))[0]
    eps255 = 8.0
    eye255 = np.eye(192, dtype=np.float32)
    plus255 = np.asarray(response_raw234(eps255 * eye255))
    minus255 = np.asarray(response_raw234(-eps255 * eye255))
    J255 = ((plus255 - minus255) / (2 * eps255)).T.astype(np.float64)

    target255 = np.array([[0.0, 3.0], [-1.9369, 0.0],
                          [0.0, 3.192], [-1.9077, 0.0]])
    primary255 = J255[[1, 2]]
    ridge255 = 0.01 * np.linalg.svd(primary255, compute_uv=False)[0] ** 2
    directions_primary255 = primary255.T @ np.linalg.solve(
        primary255 @ primary255.T + ridge255 * np.eye(2),
        target255[[1, 2]]
    )
    ridge_full255 = 0.03 * np.linalg.svd(J255, compute_uv=False)[0] ** 2
    directions_full255 = J255.T @ np.linalg.solve(
        J255 @ J255.T + ridge_full255 * np.eye(4), target255
    )
    print("Sensor order: cuff, finger, arterial, lab")
    print("Jacobian singular values:", np.linalg.svd(J255, compute_uv=False))
    for label255, directions255 in (
        ("primary", directions_primary255),
        ("full", directions_full255),
    ):
        tests255 = np.stack([
            sign * directions255[:, drug]
            for drug in (0, 1) for sign in (-1.0, 1.0)
        ]).astype(np.float32)
        shifts255 = np.asarray(response_raw234(tests255)) - zero255
        print(label255, "direction norms:", np.linalg.norm(directions255, axis=0))
        print(label255, "minus/plus one-unit sensor shifts:\n",
              np.round(shifts255, 4))
finally:
    r233_raw = saved_raw255

with Path("/work/notes.md").open("a") as log255:
    log255.write(
        "\nPython run 255; forwards 20. Verified factual dose/latent token "
        "construction and recalibrated total raw-state directions by "
        "central finite differences around original factual states. "
        "No model gradients or weight changes.\n"
    )
```