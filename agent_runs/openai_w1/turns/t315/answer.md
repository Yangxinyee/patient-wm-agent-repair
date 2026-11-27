The cached attention helper uses the expected split key/value layout, but I still need to verify its slot residuals and time decay. I’ll compare those directly with the reconstructed inputs without changing weights. Near-zero reconstruction error would justify rebuilding the fitting cache; a remaining discrepancy would mean the replay still cannot support an edit.

```python
import numpy as np, torch
import torch.nn.functional as F
from pathlib import Path

def np276(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("=== REPLAY SETTINGS ===")
print("attention layout:", rx_layout)
print("picked indices:", np276(rx_pick)[:8])
print("dt seconds:", np276(rx_dt)[:8])
print("previous indices:", rr_prev4[np276(rx_pick).astype(int)][:8])
for n in ["core.prelude.0.attn.q.weight", "core.prelude.0.attn.kv.weight",
          "core.prelude.0.attn.out.weight", "core.slots"]:
    cur = np276(load(n))
    print(n, "rx_W mismatch", float(np.sqrt(np.mean((np276(rx_W[n])-cur)**2))),
          "weights258 mismatch", float(np.sqrt(np.mean((np276(weights258[n])-cur)**2))))

print("\n=== RELEVANT CACHE NAMES ===")
for n in sorted(list(globals())):
    o = globals()[n]
    if isinstance(o, (np.ndarray, torch.Tensor)) and tuple(o.shape) in [
        (5215, 7, 192), (5215, 4, 192), (5215,), (48, 7, 192)]:
        if not re.search(r"(Es|feedback|err|error|mask|hold|train|valid|count|rate|resid|dose)", n, re.I):
            print(n, tuple(o.shape))

pick276 = np276(rx_pick).astype(int)
hours276 = np.asarray(rr_state4["hour"], dtype=float)
prev276 = np.asarray(rr_prev4, dtype=int)
dt276 = np.zeros(len(hours276), dtype=np.float32)
validprev276 = prev276 >= 0
dt276[validprev276] = ((hours276[validprev276] -
                        hours276[prev276[validprev276]]) * 3600).astype(np.float32)
hflow276 = torch.as_tensor(rr_prevh4, dtype=torch.float32) * torch.exp(
    -torch.as_tensor(dt276)[:, None, None] *
    torch.exp(load_original("core.flow.a"))[None, None, :])
slots276 = hflow276 + load_original("core.slots")
ns276 = F.layer_norm(slots276, (192,),
                    load_original("core.prelude.0.norm1.weight"),
                    load_original("core.prelude.0.norm1.bias")).numpy()

print("\n=== SLOT ALIGNMENT ===")
print("recomputed dt versus rx_dt RMS:",
      float(np.sqrt(np.mean((dt276[pick276]-np276(rx_dt))**2))))
print("recomputed hflow versus rx_hflow RMS:",
      float(np.sqrt(np.mean((hflow276[pick276].numpy()-np276(rx_hflow))**2))))
print("recomputed slots versus raw270 first four RMS:",
      float(np.sqrt(np.mean((slots276[pick276].numpy()-np276(raw270)[:, :4])**2))))
print("slots258 versus corrected slots RMS:",
      float(np.sqrt(np.mean((np276(slots258)-slots276.numpy())**2))))
print("normalized cached versus corrected slots RMS:",
      float(np.sqrt(np.mean((np276(ns258)-ns276)**2))))
print("slot RMS: cached / corrected",
      float(np.sqrt(np.mean(np276(slots258)**2))),
      float(np.sqrt(np.mean(slots276.numpy()**2))))
print("state beginning:", rr_keys4[:3], "previous", prev276[:3])

print("\n=== RUN269 NOTE ===")
s276 = Path("/work/notes.md").read_text()
m276 = s276.rfind("## Run 269")
print(s276[m276:m276+950])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 276\nNumerically auditing time decay, slot residuals, attention layout and current-versus-cached weights. Built fresh hflow/slot tokens from exported original states. No weights changed. Counts: 276 Python runs, 29 forwards.\n")
```