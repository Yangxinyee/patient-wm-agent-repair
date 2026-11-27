The earlier posterior pilot barely changed factual predictions or dose rollouts, which suggests its signal was either too small or not reaching the state effectively. The action decoder works reasonably across the tested factors, though insulin x3 is less accurate. I’ll measure the pilot’s latent dose contrasts directly to distinguish a decoding failure from a transport failure.

```python
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

torch.set_num_threads(8)

def tensor153(x):
    return torch.as_tensor(x, dtype=torch.float32)

post_original153 = {
    n: tensor153(load_original(n))
    for n in ("z_post.0.weight", "z_post.0.bias",
              "z_post.2.weight", "z_post.2.bias")
}
post_pilot153 = {n: tensor153(v) for n, v in pilot_weights_53.items()}

def posterior153(ctx, e, u, weights):
    x = torch.cat((tensor153(ctx), tensor153(e), tensor153(u)), dim=-1)
    hidden = F.gelu(F.linear(x, weights["z_post.0.weight"],
                            weights["z_post.0.bias"]))
    return F.linear(hidden, weights["z_post.2.weight"],
                    weights["z_post.2.bias"])[..., :32]

print("Posterior shapes:", {n: tuple(v.shape) for n, v in post_original153.items()})
print("Action case alignment RMSE:",
      float(np.sqrt(np.mean((np.asarray(u81) - np.asarray(U103[:len(idx81)]))**2))))

sample153 = np.arange(0, len(idx81), 4)
ctx153 = tensor153(ctx81)[sample153]
e153 = tensor153(e81)[sample153]
h153 = tensor153(h81)[sample153]
u_cases153 = tensor153(U103).reshape(len(scenarios103), len(idx81), 192)[:, sample153]
y_cases153 = np.asarray(YY103).reshape(len(scenarios103), len(idx81), 4)[:, sample153]

z_original153 = torch.stack([
    posterior153(ctx153, e153, u, post_original153) for u in u_cases153
])
z_pilot153 = torch.stack([
    posterior153(ctx153, e153, u, post_pilot153) for u in u_cases153
])

print("\nPilot factual latent change:",
      float(torch.sqrt(torch.mean((z_pilot153[0] - z_original153[0])**2))))
print("Pilot factual latent SD per dimension:",
      np.round(z_pilot153[0].std(0).numpy(), 5))
print("\nScenario contrasts: original / pilot latent RMS")
for j, scenario in enumerate(scenarios103):
    print(scenario,
          float(torch.sqrt(torch.mean((z_original153[j]-z_original153[0])**2))),
          float(torch.sqrt(torch.mean((z_pilot153[j]-z_pilot153[0])**2))))

zinw153 = tensor153(load_original("z_in.weight"))
zinb153 = tensor153(load_original("z_in.bias"))
normw153 = tensor153(load_original("core.prelude.0.norm1.weight"))
normb153 = tensor153(load_original("core.prelude.0.norm1.bias"))
latent_original153 = F.layer_norm(
    F.linear(z_original153, zinw153, zinb153),
    (192,), normw153, normb153
).numpy()
latent_pilot153 = F.layer_norm(
    F.linear(z_pilot153, zinw153, zinb153),
    (192,), normw153, normb153
).numpy()

print("\nNormalized latent factual difference RMS:",
      float(np.sqrt(np.mean((latent_pilot153[0]-latent_original153[0])**2))))
print("Normalized latent dose contrast RMS:")
for j, scenario in enumerate(scenarios103):
    print(scenario, float(np.sqrt(np.mean(
        (latent_pilot153[j]-latent_pilot153[0])**2))))

print("\nRetained calibration arrays:")
for name in ("directions136", "Jview105", "Jvalue105", "value_directions105",
             "repair_policy_w", "policy_fits148", "policy_coeff148"):
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, {k: getattr(v, "shape", type(v).__name__) for k,v in obj.items()})
    elif obj is not None:
        print(name, getattr(obj, "shape", type(obj).__name__))

print("\nForward 10 files:", [p.name for p in Path("/work/fwd_10").iterdir()])
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\nPython run 153; forwards 10. Reconstructed original and "
        "saved gated-posterior pilot on aligned factual and scaled "
        "action cases, measuring latent contrasts before transport. "
        "Previous pilot rollout failure must be distinguished from "
        "decoder failure. Current five zero-payload transport edits "
        "remain unchanged; saved posterior pilot was not installed.\n"
    )
```