On this subset, the insulin doubling prediction is about −7.64 versus the record-derived equal-stay target of −13.19; halving is much closer. I will screen three unsaved edits that amplify only the insulin innovation while lowering its gate. A factual-safe candidate would justify a rollout test; it must still improve doubling without excessive halving.

```python
import time
import numpy as np
import torch

r244_current = {n: torch.as_tensor(load(n), dtype=torch.float32).clone() for n in NAMES}
r244_saved_globals = {k: globals()[k] for k in
                      ("rW", "repair_bank", "rp_linear", "reference_bank125")}
r244_candidates = {}
r244_results = {}
r244_base_nll = float(np.mean(losses_o232))
r244_expected_n = len(baseline_losses194)

def r244_find_losses(result):
    if isinstance(result, (np.ndarray, torch.Tensor)):
        a = np.asarray(result.detach().cpu() if isinstance(result, torch.Tensor) else result)
        if a.ndim == 1 and len(a) == r244_expected_n:
            return a
    if isinstance(result, dict):
        for v in result.values():
            found = r244_find_losses(v)
            if found is not None:
                return found
    if isinstance(result, (list, tuple)):
        for v in result:
            found = r244_find_losses(v)
            if found is not None:
                return found
    return None

print("Original cached CPU NLL:", r244_base_nll)
for pair in ((96, 184), (89, 156)):
    cols = r244_current["z_post.2.weight"][:, list(pair)]
    print("Posterior pair", pair, "column sum norm", float((cols[:, 0] + cols[:, 1]).norm()),
          "column difference norm", float((cols[:, 0] - cols[:, 1]).norm()))

r244_start = time.monotonic()
try:
    for alpha, total_gate_shift in ((1.0, 26.0), (1.25, 28.0),
                                    (1.5, 30.0), (1.5, 34.0)):
        candidate = dict(r244_current)
        w = r244_current["z_post.0.weight"].clone()
        b = r244_current["z_post.0.bias"].clone()
        w[96] = w[184] + alpha * (w[96] - w[184])
        b[96] = b[184] + alpha * (b[96] - b[184])
        b[[96, 184]] -= total_gate_shift - 26.0
        candidate["z_post.0.weight"] = w
        candidate["z_post.0.bias"] = b
        key = (alpha, total_gate_shift)
        r244_candidates[key] = candidate
        result = factual_eval194(candidate)
        losses = r244_find_losses(result)
        if losses is None:
            print("Unrecognized factual result structure:",
                  [(type(v).__name__, getattr(v, "shape", None)) for v in result]
                  if isinstance(result, (tuple, list)) else type(result).__name__)
            break
        nll = float(np.mean(losses))
        r244_results[key] = {"nll": nll, "delta": nll - r244_base_nll}
        print(key, r244_results[key], "elapsed", round(time.monotonic() - r244_start, 1))
finally:
    for k, v in r244_saved_globals.items():
        globals()[k] = v

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 244\n"
            "Screened UNSAVED insulin innovation-only gains with paired-gate thresholds "
            "(1,26), (1.25,28), (1.5,30), (1.5,34); B_in and structural compensation unchanged. "
            f"CPU factual results: {r244_results}. CPU globals restored in finally. "
            "No saved weights changed. GPU factual and rollout validation remain necessary.\n")
```