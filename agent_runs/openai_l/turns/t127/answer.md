Direct attention amplification looks impractical: the fitted routes required enormous changes and still did not match the sensor targets well. I’ll instead test whether a small signal in an underused state direction can be read by the value head. That would support a separate correction channel; a noisy or inconsistent readout would refute it.

```python
import numpy as np
import torch
from scipy.linalg import solve

linear_backup118 = linear63
captured_pre118 = []
def linear_capture118(x, prefix):
    if prefix == "core.c_out":
        captured_pre118.append(x.detach().numpy().copy())
    return linear_backup118(x, prefix)

try:
    linear63 = linear_capture118
    for start118 in range(0, len(valid116), 128):
        ids118 = valid116[start118:start118+128]
        cc118, hf118, mm118, aa118, mask118 = bundle_inputs74(ids118)
        zz118 = mlp63(torch.cat(
            [cc118, e_valid60[ids118], u_valid60[ids118]], dim=-1),
            "z_post")[:, :32]
        transition64(hf118, e_valid60[ids118], u_valid60[ids118],
                     zz118, mm118, aa118, mask118)
finally:
    linear63 = linear_backup118
Hpre118 = np.concatenate(captured_pre118)
Hpost118 = sx63[valid116, 192:960].reshape(-1, 4, 192).numpy()
hp118 = Hpre118.reshape(-1, 192).astype(np.float64)
hs118 = Hpost118.reshape(-1, 192).astype(np.float64)
Cpre118 = hp118.T @ hp118 / len(hp118)
Cstate118 = hs118.T @ hs118 / len(hs118)
estate118, Ustate118 = np.linalg.eigh(Cstate118)
Rbase118 = Ustate118[:, :20]
print("state second-moment smallest:", estate118[:20])

xf118 = input_fact96.reshape(-1, 192).numpy().astype(np.float64)
xmean118 = xf118.mean(0)
xfc118 = xf118 - xmean118
Chead118 = xfc118.T @ xfc118 / len(xfc118)
print("head covariance smallest:", np.linalg.eigvalsh(Chead118)[:12])

state_rows118 = sx63[ids93[positions93]]
hrows118 = state_rows118[:, 192:960].reshape(-1, 4, 192)
epsrows118 = epsread73[:, rows93]
Jhead118 = []
Jrows118 = []
for j118 in range(Rbase118.shape[1]):
    r118 = torch.as_tensor(Rbase118[:, j118], dtype=torch.float32)
    xp118 = []
    for sign118 in (-1.0, 1.0):
        hh118 = hrows118 + sign118*r118[None, None]
        oo118 = norm63(hh118.mean(1), "core.out_norm")
        cc118, _ = context64(oo118, hh118, dtread73[rows93])
        xp118.append(head_inputs96(
            cc118, const_train67[rows93], epsrows118).numpy())
    derivative118 = (xp118[1] - xp118[0])/2.0
    Jrows118.append(derivative118.mean(0))
    Jhead118.append(derivative118.mean((0, 1)))
Jhead118 = np.asarray(Jhead118, dtype=np.float64)
Jrows118 = np.asarray(Jrows118, dtype=np.float64)

CinvJ118 = solve(Chead118 + np.eye(192)*1e-7,
                 Jhead118.T, assume_a="pos")
score118 = Jhead118 @ CinvJ118
escore118, Uscore118 = np.linalg.eigh(score118)
rotation118 = Uscore118[:, -4:]
R118 = Rbase118 @ rotation118
Jselected118 = rotation118.T @ Jhead118
Vsignal118 = solve(Chead118 + np.eye(192)*1e-7,
                   Jselected118.T, assume_a="pos")
Vsignal118 = Vsignal118 @ np.linalg.inv(Jselected118 @ Vsignal118)
responses118 = np.einsum(
    "krd,kl,dm->rlm", Jrows118, rotation118, Vsignal118)

print("private-channel detectability scores:", escore118[-8:])
print("state baseline RMS:", np.sqrt(np.diag(R118.T @ Cstate118 @ R118)))
print("head baseline SD:", np.sqrt(np.diag(Vsignal118.T @ Chead118 @ Vsignal118)))
for name118 in np.unique(sensor_codes110):
    rm118 = sensor_codes110 == name118
    print(name118, "mean response matrix",
          responses118[rm118].mean(0).round(3),
          "diagonal response SD",
          np.std(np.diagonal(responses118[rm118], axis1=1, axis2=2), axis=0).round(3))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 118 / private-state readout feasibility\n"
            "Direct attention routes require norms≈15,675/34,276 and are rejected. "
            "Captured original pre-coda states and tested four low-variance state channels "
            "with covariance-penalized value-input readouts. State baseline RMS=" +
            repr(np.sqrt(np.diag(R118.T @ Cstate118 @ R118)).tolist()) +
            "; head baseline SD=" +
            repr(np.sqrt(np.diag(Vsignal118.T @ Chead118 @ Vsignal118)).tolist()) +
            ". No weights edited.\n")
```