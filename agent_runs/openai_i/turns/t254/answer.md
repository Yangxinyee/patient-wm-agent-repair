The earlier safe head edit only moved insulin ×2 from −1.864 to −2.250 and barely changed norepinephrine’s response. The new route may offer stronger control, but its neutral version already fails the factual limit. I’ll identify its reserved response coordinates and recover the persistence evidence before fitting a correction. Stable local responses across doses would support proceeding; saturation or substantial leakage would argue against it.

```python
import dis, types, numpy as np, torch, re
from pathlib import Path

print("SELECTED DYNAMICS / IDENTIFICATION NOTES")
for b in re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M):
    if re.search(r"Python (6|8|68|191|192|221|222)\b", b[:160]):
        print(b[:650])

print("\nROUTING GEOMETRY")
for key in ["private_cols222", "hs222", "signal_scale222", "slope222",
            "gate222", "axes222", "constant222", "baseW222", "baseb222"]:
    obj = globals().get(key)
    if obj is None:
        continue
    arr = obj.detach().cpu().numpy() if torch.is_tensor(obj) else np.asarray(obj)
    print(key, "shape", arr.shape, "norm", float(np.linalg.norm(arr)),
          "values", arr.reshape(-1)[:8])
print("Reserved output column norms",
      np.linalg.norm(np.asarray(out222)[:, np.asarray(private_cols222, dtype=int)], axis=0))
print("W distance from original / installed")
for k in ["z_in.weight", "core.prelude.0.attn.out.weight"]:
    print(k, float(torch.norm(W[k] - torch.as_tensor(load_original(k)))),
          float(torch.norm(W[k] - torch.as_tensor(load(k)))))

print("\nREPLAY / CORE CONTRACTS")
for fnname in ["core_step136", "call_core161", "linear136", "mlp136",
               "encode_actions168", "replay_action_token216", "batch_replay182"]:
    fn = globals().get(fnname)
    if callable(fn):
        c = fn.__code__
        print(fnname, "args", c.co_varnames[:c.co_argcount],
              "locals", c.co_varnames[c.co_argcount:],
              "globals", c.co_names)
print("\nFIRST CACHED ARGUMENT")
print([(tuple(x.shape), str(x.dtype)) if torch.is_tensor(x) else type(x).__name__
       for x in args_cache182[0]])
print("ids182", ids182[:10], "range", (min(ids182), max(ids182)))
for key in ["input196", "all_z201", "new_z209", "private_z209", "context",
            "all_e196", "all_u196", "reg_terms182", "sensor_order214",
            "centres214", "gap_seconds182"]:
    obj = globals().get(key)
    if obj is not None:
        print(key, tuple(obj.shape) if hasattr(obj, "shape") else type(obj).__name__)
print("\nACTION REPLAY INTERCEPT")
for c in replay_action_token216.__code__.co_consts:
    if isinstance(c, types.CodeType):
        dis.dis(c)

with open("/work/notes.md", "a") as f227:
    f227.write("\n## Python 227 — response-coordinate and replay inspection\n"
               "- Safe fallback remains weak; small route remains uncertified and over the factual limit.\n"
               "- Recovering private routing coordinates and forward-only replay inputs before calibration.\n"
               "- No new weights changed.\n")
```