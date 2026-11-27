Another unmatched parenthesis prevented execution, so the edit is unchanged. I’ll rerun the diagnostic with simpler expressions. The key test remains whether edited doses activate the correction and whether that correction reaches the recurrent state.

```python
import numpy as np
import torch
import torch.nn.functional as F

print("REPAIR DIRECTIONS")
print("latent direction norms:", np.linalg.norm(repair_zdirs, axis=0))
print("selected hidden units:", repair_units)
print("pilot strengths:", repair_pilot_strength)
for drug, item in repair_centering.items():
    if isinstance(item, dict):
        description = {
            k: tuple(v.shape) if hasattr(v, "shape") else repr(v)[:150]
            for k, v in item.items()
        }
    else:
        description = repr(item)[:300]
    print("centering", drug, description)

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
        rms = float(np.sqrt(np.mean(array ** 2)))
        maximum = float(np.max(np.abs(array)))
        print(name, array.shape, "rms", rms, "max", maximum)

def direct_posterior_diagnostic(inputs, weights):
    x = torch.as_tensor(diagnostic_array(inputs), dtype=torch.float32)
    w0 = torch.as_tensor(diagnostic_array(weights["z_post.0.weight"]))
    b0 = torch.as_tensor(diagnostic_array(weights["z_post.0.bias"]))
    w2 = torch.as_tensor(diagnostic_array(weights["z_post.2.weight"]))
    b2 = torch.as_tensor(diagnostic_array(weights["z_post.2.bias"]))
    pre = F.linear(x, w0, b0)
    hidden = F.gelu(pre)
    post = F.linear(hidden, w2, b2)
    return pre.numpy(), hidden.numpy(), post.numpy()

_diag_input60 = globals().get("r46_input", posterior_inputs)
_diag_original60 = {name: load_original(name) for name in repair_names}
_orig_pre60, _orig_hidden60, _orig_post60 = direct_posterior_diagnostic(
    _diag_input60, _diag_original60)
_edit_pre60, _edit_hidden60, _edit_post60 = direct_posterior_diagnostic(
    _diag_input60, pilot_weights_53)

print("\nPOSTERIOR PILOT ON RETAINED INPUTS:", _orig_post60.shape)
for unit in repair_units:
    unit = int(unit)
    original_quantiles = np.quantile(_orig_pre60[:, unit], [0, .1, .5, .9, 1])
    edited_quantiles = np.quantile(_edit_pre60[:, unit], [0, .1, .5, .9, 1])
    nonzero_fraction = float(np.mean(np.abs(_edit_hidden60[:, unit]) > 1e-5))
    print("unit", unit, "original", original_quantiles,
          "edited", edited_quantiles, "nonzero fraction", nonzero_fraction)

_delta_post60 = _edit_post60 - _orig_post60
for label, section in (("mean", slice(0, 32)), ("log-sd", slice(32, 64))):
    delta = _delta_post60[:, section]
    rms = float(np.sqrt(np.mean(delta ** 2)))
    maximum = float(np.max(np.abs(delta)))
    print(label, "delta rms/max:", rms, maximum)

print("\nDECODED FACTUAL ACTION QUANTILES")
for name in ("repair_actual_pred", "repair_decoded"):
    array = diagnostic_array(globals()[name])
    print(name, np.quantile(array, [0, .1, .5, .9, 1], axis=0))

print("\nACTION HELPER SOURCE")
try:
    print(inspect.getsource(repair_action_vectors)[:3000])
except Exception as exc:
    print(type(exc).__name__, str(exc))
    print("helper bytecode names:", repair_action_vectors.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 60; forwards 5. Run 59 was a parse-time failure; "
            "no changes. Repeating posterior-gate and latent-path diagnostics.\n")
```