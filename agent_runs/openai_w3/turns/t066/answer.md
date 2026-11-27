The pilot uses very large input coefficients, yet its rollout effect is tiny. That points to either a gating/scaling problem or a weak latent-to-state pathway. I’ll measure the gate activations, posterior changes, and downstream weights. Large posterior changes with tiny rollout changes would refute the assumption that this latent channel is an effective repair route.

```python
import numpy as np
import torch
import torch.nn.functional as F

print("REPAIR DIRECTIONS")
print("latent direction norms:", np.linalg.norm(repair_zdirs, axis=0))
print("selected hidden units:", repair_units)
print("pilot strengths:", repair_pilot_strength)
for drug, item in repair_centering.items():
    print("centering", drug,
          {k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:200])
           for k, v in item.items()} if isinstance(item, dict) else repr(item)[:300])

print("\nINPUT / STATE METADATA")
for name in ("r46_input", "repair_H", "repair_context", "e_all", "u_all",
             "repair_actual", "repair_actual_pred", "repair_decoded",
             "state_records", "S", "Sfile", "r45_sensor_directions"):
    obj = globals().get(name)
    if hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "keys", list(obj.keys())[:15])
    elif hasattr(obj, "files"):
        print(name, "npz keys", obj.files)
    else:
        print(name, type(obj).__name__)

print("\nLATENT PATH WEIGHTS")
for name in NAMES:
    if "z_in" in name or "z_head" in name:
        array = diagnostic_array(load_original(name))
        print(name, array.shape, "rms", float(np.sqrt(np.mean(array ** 2))),
              "max", float(np.max(np.abs(array)))))

def direct_posterior_diagnostic(inputs, weights):
    x = torch.as_tensor(diagnostic_array(inputs), dtype=torch.float32)
    w0 = torch.as_tensor(diagnostic_array(weights["z_post.0.weight"]))
    b0 = torch.as_tensor(diagnostic_array(weights["z_post.0.bias"]))
    w2 = torch.as_tensor(diagnostic_array(weights["z_post.2.weight"]))
    b2 = torch.as_tensor(diagnostic_array(weights["z_post.2.bias"]))
    pre = F.linear(x, w0, b0)
    hidden = F.gelu(pre)
    return pre.numpy(), hidden.numpy(), F.linear(hidden, w2, b2).numpy()

_diag_input59 = globals().get("r46_input", posterior_inputs)
_diag_original59 = {name: load_original(name) for name in repair_names}
_orig_pre59, _orig_hidden59, _orig_post59 = direct_posterior_diagnostic(
    _diag_input59, _diag_original59)
_edit_pre59, _edit_hidden59, _edit_post59 = direct_posterior_diagnostic(
    _diag_input59, pilot_weights_53)

print("\nPOSTERIOR PILOT ON RETAINED INPUTS:", _orig_post59.shape)
for unit in repair_units:
    unit = int(unit)
    print("unit", unit,
          "original pre quantiles", np.quantile(_orig_pre59[:, unit], [0, .1, .5, .9, 1]),
          "edited pre quantiles", np.quantile(_edit_pre59[:, unit], [0, .1, .5, .9, 1]),
          "edited hidden nonzero fraction",
          float(np.mean(np.abs(_edit_hidden59[:, unit]) > 1e-5)))
_delta_post59 = _edit_post59 - _orig_post59
print("mean delta: rms/max",
      float(np.sqrt(np.mean(_delta_post59[:, :32] ** 2))),
      float(np.max(np.abs(_delta_post59[:, :32]))))
print("log-sd delta: rms/max",
      float(np.sqrt(np.mean(_delta_post59[:, 32:] ** 2))),
      float(np.max(np.abs(_delta_post59[:, 32:]))))

print("\nDECODED FACTUAL ACTION QUANTILES")
for name in ("repair_actual_pred", "repair_decoded"):
    array = diagnostic_array(globals()[name])
    print(name, np.quantile(array, [0, .1, .5, .9, 1], axis=0))

print("\nACTION HELPER SOURCE")
try:
    print(inspect.getsource(repair_action_vectors)[:3500])
except Exception as exc:
    print(type(exc).__name__, str(exc))
    print("helper bytecode names:", repair_action_vectors.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 59; forwards 5. Diagnosing posterior gate activations, "
            "mean-vs-logsd changes, latent direction scaling, and core.z_in/z_head "
            "path strength before changing pilot magnitude.\n")
```