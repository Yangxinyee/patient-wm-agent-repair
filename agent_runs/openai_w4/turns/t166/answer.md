The provisional record-based sustained targets are much larger than the current rollouts: approximately −13 to −16 mg/dL for insulin doubling and +3.6 to +3.9 mmHg for norepinephrine doubling. They remain assumption-dependent estimates, not supplied truth. I’ll now measure channel responses to actual dose changes; near-proportional responses would refute dose compression as the main explanation.

```python
import numpy as np
import torch

rW_before147 = rW
x_before147 = repair_bank["x"]
norm_before147 = rp_norm
weights_current147 = {
    k: torch.as_tensor(load(k), dtype=torch.float32).clone() for k in NAMES
}
decoder147 = torch.as_tensor(chosen_channel116["decoder"], dtype=torch.float32)
channel_probe147 = {}

try:
    rW = weights_current147
    repair_bank["x"] = reference_x125.copy()

    for drug147 in ["insulin", "norepinephrine"]:
        fixture147 = controlled_action_repair[drug147]
        indices147 = np.asarray(fixture147["indices"], dtype=int)
        records147 = fixture147["records"]
        rate147 = records147["rate"].to_numpy(dtype=np.float32)
        dose147 = records147["dose"].to_numpy(dtype=np.float32)
        value147 = records147["value"].to_numpy(dtype=np.float32)
        results147 = {}

        for factor147 in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]:
            atoms147 = atom_exact65(
                drug147, value147, rate147 * factor147, dose147 * factor147
            )
            u147 = rp_action_encoder(atoms147)
            capture147 = {}

            def norm_capture147(*args, **kwargs):
                result = norm_before147(*args, **kwargs)
                prefix = args[1] if len(args) > 1 else kwargs.get("prefix", "")
                if prefix == "core.prelude_norm":
                    capture147["ep"] = result.detach().clone()
                return result

            rp_norm = norm_capture147
            transition73(indices147, action_override=u147)
            rp_norm = norm_before147
            if "ep" not in capture147:
                raise RuntimeError("Prelude capture failed; do not infer a dose response.")
            q147 = (capture147["ep"] @ decoder147).mean(dim=1).cpu().numpy()
            results147[factor147] = {"q": q147, "u": u147.detach().clone()}

        print("\nDRUG:", drug147, "n:", len(indices147))
        u_ref147 = fixture147["results"][1.0]["u"]
        print("reconstructed factual action RMS:",
              float(torch.sqrt(torch.mean((results147[1.0]["u"] - u_ref147) ** 2))))
        q_ref147 = results147[1.0]["q"]
        print("factual q quantiles:", np.quantile(q_ref147, [0, .1, .5, .9, 1]).round(4))
        for factor147 in [0.0, 0.5, 2.0, 3.0, 4.0]:
            dr147 = (factor147 - 1.0) * rate147
            dq147 = results147[factor147]["q"] - q_ref147
            slope147 = float(dr147 @ dq147 / max(dr147 @ dr147, 1e-12))
            print("factor", factor147,
                  "mean dq", round(float(dq147.mean()), 5),
                  "dose-to-channel slope", round(slope147, 5),
                  "dq/dr quantiles",
                  np.quantile(dq147 / np.where(dr147 != 0, dr147, np.nan),
                              [.1, .5, .9]).round(4))
        channel_probe147[drug147] = {
            "indices": indices147, "rate": rate147,
            "results": results147
        }
finally:
    rp_norm = norm_before147
    rW = rW_before147
    repair_bank["x"] = x_before147

with open("/work/notes.md", "a") as f147:
    f147.write(
        "\n## Run 147\n"
        "Probed actual action encodings and decoded prelude channel under dose multipliers "
        "0, 0.5, 1, 2, 3 and 4, using fixed factual contexts. "
        "This is a model-mechanism diagnostic, not causal evidence. No weights changed.\n"
    )
```