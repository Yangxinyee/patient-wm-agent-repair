The persistence analysis supports a multi-hour insulin effect, so matching only the next reading would be insufficient. The private route is still unvalidated. I’ll build a forward-only replay of that route and check it against the original replay before using it to calibrate response weights. A mismatch would invalidate the local calibration.

```python
import numpy as np, torch, re
from pathlib import Path

torch.set_num_threads(8)
for b in re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M):
    if re.search(r"Python 8\b", b[:120]):
        print("IMPULSE EVIDENCE\n", b[:3600])

route228 = {k: np.array(v, copy=True) for k, v in route222.items()}
normcols228 = np.linalg.norm(route228["core.prelude.0.attn.out.weight"], axis=0)
response_cols228 = np.flatnonzero(normcols228 < 1e-7)
print("ZERO ATTENTION OUTPUT COLUMNS", response_cols228)
print("SMALLEST COLUMN NORMS",
      [(int(j), float(normcols228[j])) for j in np.argsort(normcols228)[:8]])
print("TOKEN SHAPES", tokens222.shape, orig_tokens222.shape)
print("RATE / INNOVATION ARRAYS")
for key in sorted(list(globals())):
    if re.search(r"(innov|gated|policy|rate_vector|residual).*?(194|196|201|209|215|222)$", key):
        obj = globals()[key]
        print(key, tuple(obj.shape) if hasattr(obj, "shape") else type(obj).__name__)

original_post228 = mlp136(torch.cat([context, e_cal161, Ubase215], 1), "z_post")[:, :32]
print("RECOMPUTED ORIGINAL POSTERIOR VS CACHE RMS",
      float(torch.sqrt(torch.mean((original_post228[ids182] - oldpost182)**2))))

def replay_route228(candidate, action_inputs=None, indices=None, batch_size=128):
    """Forward-only one-bundle replay, using candidate weights and posterior."""
    if action_inputs is None:
        action_inputs = Ubase215
    action_inputs = torch.as_tensor(action_inputs, dtype=torch.float32)
    ix = np.arange(n182) if indices is None else np.asarray(indices, dtype=int)
    groups = {}
    for p, j in enumerate(ix):
        groups.setdefault(args_cache182[int(j)][5].shape[1], []).append((p, int(j)))
    retained = {k: W[k] for k in candidate}
    try:
        for k, v in candidate.items():
            W[k] = torch.as_tensor(v, dtype=torch.float32)
        posterior = mlp136(torch.cat([context, e_cal161, action_inputs], 1), "z_post")[:, :32]
        result_h = torch.empty(len(ix), 4, 192)
        result_o = torch.empty(len(ix), 192)
        for pairs in groups.values():
            for offset in range(0, len(pairs), batch_size):
                part = pairs[offset:offset + batch_size]
                positions = [p for p, j in part]
                js = [j for p, j in part]
                packed = [torch.cat([args_cache182[j][k] for j in js], 0)
                          for k in range(7)]
                allids = np.asarray(ids182)[js]
                packed[3] = action_inputs[allids]
                packed[4] = posterior[allids]
                hh, oo = call_core161(tuple(packed), "key")
                result_h[positions] = hh
                result_o[positions] = oo
        return result_h, result_o
    finally:
        for k, v in retained.items():
            W[k] = v

check_h228, check_o228 = replay_route228({})
print("ORIGINAL REPLAY VS EXISTING REPLAY RMS",
      float(torch.sqrt(torch.mean((check_h228 - base_h214)**2))),
      float(torch.sqrt(torch.mean((check_o228 - base_o214)**2))))
route_h228, route_o228 = replay_route228(route228)
route_means228 = readout214(route_h228, route_o228)
base_means228 = readout214(check_h228, check_o228)
print("ROUTE SENSOR MEAN SHIFT",
      np.asarray(route_means228 - base_means228).mean(axis=0))
print("REPLAY RESULT SHAPES", route_h228.shape, route_o228.shape,
      np.asarray(route_means228).shape)

with open("/work/notes.md", "a") as f228:
    f228.write("\n## Python 228 — forward-only route replay\n"
               "- Recovered record-derived impulse evidence and identified unused response-output columns.\n"
               "- Built candidate-weight replay with recomputed posterior; compared it with retained original replay.\n"
               "- No additional edits installed. Local replay is calibration evidence, not rollout validation.\n")
```