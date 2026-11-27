The finite-dose test found a transport failure: some norepinephrine rates produced private-state signals orders of magnitude larger than nearby rates, even though the encoder’s rate signal stayed linear. That points to routing, not drug biology. I’ll inspect the routing gate and look for a dose-independent carrier that selects the action token reliably across bins.

```python
import inspect
import numpy as np
import torch

print("=== PRELUDE TENSOR NAMES ===")
print([n for n in NAMES if n.startswith("core.prelude.0.")])
for fn_name in ["wm_layer", "wm_attention", "wm_norm", "wm_set",
                "exact_u162", "action_vectors4"]:
    fn = globals()[fn_name]
    print(fn_name, inspect.signature(fn), "constants:",
          [v for v in fn.__code__.co_consts
           if isinstance(v, (str, int, float, tuple))][:35])

print("=== CURRENT PRIVATE CHANNEL PARAMETERS ===")
for name in ["encode.norm.weight", "encode.norm.bias",
             "encode_action.norm.weight", "encode_action.norm.bias",
             "core.slots", "core.out_norm.weight", "core.out_norm.bias"]:
    w = torch.as_tensor(load(name))
    print(name, w[..., idx182].numpy().round(7).tolist())
print("gate vector nonzeros:",
      [(int(i), float(gate_vector[i])) for i in
       np.flatnonzero(np.abs(np.asarray(gate_vector)) > 1e-9)][:24])
print("gate bias/sign:", gate_bias, gate_sign)
print("private flow", torch.as_tensor(load("core.flow.a"))[idx182].numpy().tolist())

print("=== RAW TOKEN CARRIER CANDIDATE ===")
carrier190 = torch.zeros(192)
carrier190[list(pairs183[0])] = 1.0
carrier190[list(pairs183[1])] = -1.0
for j in [3, 31, 57, 59, 61]:
    _, loc = capture177(wm_step_boundary176, j)
    tokens = loc["tokens"].detach().reshape(7, 192)
    print("bundle", j,
          "raw carrier projection", (tokens @ carrier190).numpy().round(7).tolist(),
          "token RMS", tokens.square().mean(1).sqrt().numpy().round(4).tolist())

print("=== SAVED-STATE AGREEMENT FOR PROBLEMATIC NE BUNDLES ===")
for j in [57, 59, 61]:
    h = wm_step_boundary176(j).detach().reshape(4, 192).numpy()
    true = np.asarray(s4["x"][j, 192:960]).reshape(4, 192)
    print(j, "private error", np.sqrt(np.mean((h[:, idx182] - true[:, idx182])**2)),
          "saved NE pair", (true[:, pairs183[1][0]] - true[:, pairs183[1][1]]).mean(),
          "CPU NE pair", (h[:, pairs183[1][0]] - h[:, pairs183[1][1]]).mean())
```
