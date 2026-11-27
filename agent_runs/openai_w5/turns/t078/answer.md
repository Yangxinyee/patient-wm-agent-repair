Fixing the boundaries removed the memory failures, and globally packed attention heads reproduce the states much better than the previous convention. Some error remains. I’ll now check whether memory includes the current bundle and how age enters attention. A convention that consistently matches held-out bundles will support recalculating the dose-response directions; otherwise, I’ll keep treating them as approximate.

```python
import numpy as np
import torch

def local_step67(j, z_batch, include_current=False, age_mode="input",
                 age_units=3600.0, age_scale=1.0):
    z_batch = torch.as_tensor(z_batch, dtype=torch.float32)
    K = len(z_batch)
    assert s4["stay_id"][j-1] == s4["stay_id"][j]
    hh = torch.from_numpy(s4["x"][j-1:j, 192:960]).reshape(1, 4, 192)
    dt = round(float(s4["hour"][j] - s4["hour"][j-1]) * 60) * 60.0
    hh = (hh * torch.exp(-dt * torch.exp(W0["core.flow.a"]))).expand(K, -1, -1)
    ee = torch.as_tensor(wm_e4[j:j+1]).expand(K, -1)
    uu = torch.as_tensor(U4[j:j+1]).expand(K, -1)
    tokens = torch.cat([
        hh + W0["core.slots"], ee[:, None], uu[:, None],
        wm_lin(z_batch, "z_in")[:, None]], dim=1)
    E = wm_norm(wm_layer(tokens, "core.prelude.0")[:, :4],
                "core.prelude_norm")
    end = j + int(include_current)
    begin = max(start_by_stay66[int(s4["stay_id"][j])], end - 128)
    memory = torch.as_tensor(wm_e4[begin:end])[None].expand(K, -1, -1)
    ages = torch.as_tensor(
        (s4["hour"][j] - s4["hour"][begin:end]) * age_units,
        dtype=torch.float32).reshape(1, -1, 1)
    age_vectors = wm_mlp(torch.log1p(ages) * age_scale,
                         "core.age").expand(K, -1, -1)
    bb = hh + E
    for layer in range(2):
        bb = wm_layer(bb, f"core.blocks.0.{layer}", memory,
                      age_vectors, age_mode)
    delta = torch.nn.functional.softplus(W0["core.delta_in"])
    abar = torch.exp(-delta * torch.exp(W0["core.a_in"]))
    new_h = abar * hh + delta * wm_lin(E, "core.b_in") + bb - (hh + E)
    return wm_layer(wm_lin(new_h, "core.c_out"), "core.coda.0")

old_kv67, old_mem67 = wm_kv_layout, wm_memory_mode
recurrence_trials67 = []
try:
    wm_kv_layout = "global"
    for memory_mode67 in ["unnormalized", "norm_then_age",
                          "age_then_norm", "age_then_plainnorm"]:
        wm_memory_mode = memory_mode67
        for current67 in [False, True]:
            for age_mode67 in ["input", "key"]:
                for age_units67 in [1.0, 60.0, 3600.0]:
                    for age_scale67 in [1.0, 0.1]:
                        hs67, os67 = [], []
                        for j67 in test_rows66:
                            h67 = local_step67(
                                int(j67), s4["x"][j67:j67+1, 960:992],
                                current67, age_mode67, age_units67, age_scale67)
                            ht67 = torch.from_numpy(
                                s4["x"][j67:j67+1, 192:960]).reshape(1, 4, 192)
                            ot67 = torch.from_numpy(s4["x"][j67:j67+1, :192])
                            hs67.append(float(((h67-ht67)**2).mean().sqrt()))
                            o67 = wm_norm(h67.mean(1), "core.out_norm")
                            os67.append(float(((o67-ot67)**2).mean().sqrt()))
                        recurrence_trials67.append({
                            "memory": memory_mode67,
                            "include_current": current67, "age_mode": age_mode67,
                            "age_units": age_units67, "age_scale": age_scale67,
                            "h": np.asarray(hs67), "out": np.asarray(os67)})
finally:
    wm_kv_layout, wm_memory_mode = old_kv67, old_mem67

recurrence_trials67.sort(key=lambda t: np.mean(t["out"]))
print("=== BEST RECURRENCE CONVENTIONS ===")
for trial67 in recurrence_trials67[:10]:
    print({k: v for k, v in trial67.items() if k not in ["h", "out"]},
          "h RMSE", np.round(trial67["h"], 5),
          "out RMSE", np.round(trial67["out"], 6))
best67 = recurrence_trials67[0]
print("Chosen diagnostic convention:", {
    k: v for k, v in best67.items() if k not in ["h", "out"]})

print("\n=== VOCAB TABLE INVENTORY ===")
for key67, value67 in VOCAB.items():
    if isinstance(value67, dict):
        print(key67, "dict", list(value67)[:15])
    elif isinstance(value67, list):
        print(key67, "list length", len(value67))
    else:
        print(key67, str(value67)[:160])
print("\n=== LOCAL ENCODER CONSTANTS ===")
for name67 in ["measurement_atom_local", "action_atom_exact"]:
    fn67 = globals()[name67]
    print(name67, "globals", fn67.__code__.co_names,
          "constants", [v for v in fn67.__code__.co_consts
                        if isinstance(v, (str, int, float, tuple))])
print("Edits:", edited())
```