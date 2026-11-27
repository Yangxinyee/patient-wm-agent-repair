The recovered lag estimates suggest insulin’s response peaks around the second hour and then decays; norepinephrine has a shorter positive response. Also, zero-valued categorical rows make the empty-encoder hypothesis testable. I’ll compare several empty-bundle encodings against the exported posterior. A near-exact match would let me compute action-pathway edits with much more confidence.

```python
import torch
import torch.nn.functional as F
import numpy as np

empty0 = torch.zeros(1, 192)
empty_encodings_repair = {
    "zero": empty0,
    "mlp2_zero": rp_norm(rp_mlp(empty0, "encode.mlp2"), "encode.norm"),
    "mlp2_size_bias": rp_norm(
        rp_mlp(rW["encode.size.bias"].reshape(1, 192), "encode.mlp2"),
        "encode.norm"
    ),
}
print("Empty-encoding RMS:", {
    k: float(v.square().mean().sqrt()) for k, v in empty_encodings_repair.items()
})

uu = torch.as_tensor(u, dtype=torch.float32)
cc = torch.as_tensor(contexts, dtype=torch.float32)
tt = torch.as_tensor(target, dtype=torch.float32)
posterior_empty_tests_repair = []
for label, ee in empty_encodings_repair.items():
    ee = ee.reshape(1, 1, 192).expand_as(uu)
    xx = torch.cat([cc, ee, uu], dim=-1)
    zz = rp_mlp(xx, "z_post")[..., :32]
    err = (zz - tt.unsqueeze(0)).square().mean(dim=(1, 2)).sqrt()
    posterior_empty_tests_repair.append((label, float(err.min()), int(err.argmin())))
print("Posterior checks:", posterior_empty_tests_repair)

print("\n=== CONTEXT CHECK ===")
fx = action_fixture_repair
sx = np.load("/work/fwd_5/states_original.npz")
ii = fx["state_index"].to_numpy(dtype=int)
prev = torch.as_tensor(sx["x"][ii - 1], dtype=torch.float32)
dsec = torch.as_tensor(
    (sx["hour"][ii] - sx["hour"][ii - 1]) * 3600, dtype=torch.float32
)
hflow = prev[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -dsec[:, None, None] * torch.exp(rW["core.flow.a"])[None, None, :]
)
ctx_new = (
    prev[:, :192] + rp_norm(hflow.mean(1), "core.out_norm")
    + rp_mlp(torch.log1p(dsec).reshape(-1, 1) / 10, "gap")
)
print("recomputed versus retained context RMSE:",
      float((ctx_new - torch.as_tensor(action_ctx_repair)).square().mean().sqrt()))
print("fixture posterior RMS:",
      float(torch.as_tensor(posterior_target_repair).square().mean().sqrt()))
note("Run 38: tested zero versus learned empty measurement encodings against isolated action posterior states; checked the pre-action context from exported states. No edits.")
```
