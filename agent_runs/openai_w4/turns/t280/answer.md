Innovation-only scaling avoids the large factual damage caused by scaling the whole repair matrix. A gain of 1.25 passes the CPU screen; 1.5 leaves little margin for GPU validation. I will test stricter gates at the larger gains. Lower factual loss would support their safety, but only a rollout test can show whether doubled doses still transmit.

```python
import numpy as np
import torch

r245_candidates = {}
r245_results = {}
r245_saved_globals = {k: globals()[k] for k in
                      ("rW", "repair_bank", "rp_linear", "reference_bank125")}

try:
    for alpha, total_gate_shift in ((1.5, 40.0), (1.5, 50.0), (1.5, 60.0),
                                    (2.0, 40.0), (2.0, 50.0), (2.0, 60.0)):
        candidate = dict(r244_current)
        w = r244_current["z_post.0.weight"].clone()
        b = r244_current["z_post.0.bias"].clone()
        w[96] = w[184] + alpha * (w[96] - w[184])
        b[96] = b[184] + alpha * (b[96] - b[184])
        b[[96, 184]] -= total_gate_shift - 26.0
        candidate["z_post.0.weight"] = w
        candidate["z_post.0.bias"] = b
        key = (alpha, total_gate_shift)
        r245_candidates[key] = candidate
        result = factual_eval194(candidate)
        losses = r244_find_losses(result)
        if losses is None:
            raise RuntimeError("Factual result did not contain the expected reading losses.")
        nll = float(np.mean(losses))
        r245_results[key] = {"nll": nll, "delta": nll - r244_base_nll}
        print(key, r245_results[key])
finally:
    for k, v in r245_saved_globals.items():
        globals()[k] = v

print("\n=== RETAINED INSULIN ACTION-EDIT FIXTURE NAMES ===")
r245_g = dict(globals())
for n, v in sorted(r245_g.items()):
    if "217" in n and not callable(v):
        print(n, type(v).__name__, getattr(v, "shape", None),
              list(v)[:8] if isinstance(v, dict) else "")

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 245\n"
            "Screened UNSAVED stricter insulin gates at innovation gains 1.5/2. "
            f"CPU factual results: {r245_results}. "
            "These are computational transmission/safety tests, not pharmacological "
            "nonlinearity estimates. CPU globals restored; submitted weights unchanged.\n")
```