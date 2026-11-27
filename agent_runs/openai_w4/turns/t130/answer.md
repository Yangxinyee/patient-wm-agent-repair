The measured state sensitivities could support a glucose-specific correction. First, I’ll save the channel-only candidate, without adding that correction, and test its factual cost. If the rewiring alone exceeds 0.005 nats, we must compensate or reject it before adding drug effects.

```python
import torch, numpy as np
channel_base117 = {
    nm: torch.as_tensor(np.asarray(ww), dtype=torch.float32).clone()
    for nm, ww in chosen_channel116["weights"].items()
}
channel_changed117 = []
for nm, ww in channel_base117.items():
    orig117 = torch.as_tensor(load_original(nm), dtype=torch.float32)
    if torch.max(torch.abs(ww - orig117)).item() > 1e-7:
        save_edit(nm, ww)
        channel_changed117.append(nm)
print("Saved channel-only candidate:", channel_changed117)
print("Decoder norm:", float(np.linalg.norm(chosen_channel116["decoder"])))
print("No outcome-response connection has been added.")
print("Current edits:", edited())
with open("/work/notes.md", "a") as nf:
    nf.write("\n## Run 117: first saved candidate\n"
             "Saved amplitude-80 explicit insulin-innovation channel (9 tensors): "
             "z_post hidden/output, z_in, and prelude attention q/kv/out weights. "
             "No B_in response connection yet. This isolates factual cost of "
             "rewiring; forward factual validation is next. "
             "Original first-100 tuning NLL was 1.59524; allowed increase 0.005.\n")
```
