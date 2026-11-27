The fivefold numeric edit did transmit dose changes and stayed within the factual-loss bound on its tested batch, but its norepinephrine response remained far too weak. I’ll measure where the prelude attention sends its weight. If it scarcely attends to the latent token, that would explain why the compensated latent carriers failed; substantial attention would instead point to suppression downstream.

```python
import numpy as np, torch
from pathlib import Path

def np205(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("=== MATCHED BUNDLE CACHE ===")
for n in sorted(list(globals())):
    o = globals()[n]
    if isinstance(o, (np.ndarray, torch.Tensor)):
        s = tuple(o.shape)
        if s and s[0] == 5215 and (n.startswith(("rr_", "rx_", "rp_", "lc_", "carrier"))):
            print(n, s)
for n in ("rr_enc1_4", "rp_gate_fits", "lc_specs", "rp_specs"):
    if n in globals() and isinstance(globals()[n], dict):
        print(n, {str(k): (tuple(v.shape) if hasattr(v, "shape") else
                          list(v.keys()) if isinstance(v, dict) else type(v).__name__)
                  for k, v in globals()[n].items()})

print("\n=== PRELUDE ATTENTION TO E / ACTION / LATENT ===")
h205 = globals().get("rr_prevh4")
e205 = globals().get("rr_e4")
u205 = globals().get("rr_u4")
z205 = globals().get("carrier200_token_original")
if all(x is not None for x in (h205, e205, u205, z205)):
    h205, e205, u205, z205 = map(np205, (h205, e205, u205, z205))
    print("source shapes:", h205.shape, e205.shape, u205.shape, z205.shape)
    if h205.shape == (5215, 4, 192) and all(x.shape == (5215, 192) for x in (e205, u205, z205)):
        ix205 = np.linspace(0, 5214, 600).astype(int)
        slots205 = np205(load_original("core.slots"))
        tokens205 = np.concatenate(
            (h205[ix205] + slots205[None],
             e205[ix205, None], u205[ix205, None], z205[ix205, None]), axis=1)
        x205 = torch.as_tensor(tokens205, dtype=torch.float32)
        nw205 = torch.as_tensor(np205(load_original("core.prelude.0.norm1.weight")))
        nb205 = torch.as_tensor(np205(load_original("core.prelude.0.norm1.bias")))
        normtokens205 = torch.nn.functional.layer_norm(x205, (192,), nw205, nb205)
        q205 = torch.nn.functional.linear(
            normtokens205, torch.as_tensor(np205(load_original("core.prelude.0.attn.q.weight"))))
        kv205 = torch.nn.functional.linear(
            normtokens205, torch.as_tensor(np205(load_original("core.prelude.0.attn.kv.weight"))))
        k205, v205 = kv205.chunk(2, dim=-1)
        q205 = q205.reshape(-1, 7, 4, 48).transpose(1, 2)
        k205 = k205.reshape(-1, 7, 4, 48).transpose(1, 2)
        a205 = torch.softmax(q205 @ k205.transpose(-1, -2) / np.sqrt(48), dim=-1)
        for j205, label205 in ((4, "e"), (5, "action"), (6, "latent")):
            vals205 = a205[:, :, :4, j205].numpy()
            print(label205, "mean by head", vals205.mean(axis=(0, 2)),
                  "quantiles", np.quantile(vals205, [0, .1, .5, .9, 1]))
        print("slot-token norm RMS:", np.sqrt(np.mean(tokens205[:, :4]**2)),
              "e/u/z RMS:", [float(np.sqrt(np.mean(tokens205[:, j]**2))) for j in (4,5,6)])
    else:
        print("Attention calculation skipped: cache shapes differ from expected.")
else:
    print("Attention calculation skipped; missing:",
          [n for n in ("rr_prevh4", "rr_e4", "rr_u4", "carrier200_token_original")
           if globals().get(n) is None])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 205\nInspecting exact matched bundle inputs and prelude attention allocation to locate the failed latent transmission. Numeric 5x remains only a saved fallback, not an active edit. Counts: 205 Python runs, 18 forwards.\n")
```