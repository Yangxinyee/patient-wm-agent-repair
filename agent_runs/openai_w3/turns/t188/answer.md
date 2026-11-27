The replay uses roughly 55 minutes between these drug bundles and the next readings. The diagnostic stopped on a tensor-copy error, not a model failure. I’ll fix that and check both state reconstruction and latent-payload decoding. Close replay agreement and dose contrasts in physical units would make calibration credible.

```python
import numpy as np, torch
import torch.nn.functional as F
zero169 = {n: tensor153(load(n)).clone() for n in edited()}
print("Saved reversible zero-payload checkpoint:", list(zero169))

r169 = prelude155(tokens_carrier155[positions155], zero169)
print("Prelude result shapes:", [tuple(x.shape) for x in r169])
print("Carrier orthogonality:", carrier154.T @ carrier154)
print("Extractor on carrier:", value_extract154.T @ carrier154)
print("Payload factual mean/SD:",
      np.mean(payload154, axis=0), np.std(payload154, axis=0))

i169 = np.asarray(indices155, dtype=int)
dt169 = tensor153((S["hour"][i169] - S["hour"][i169-1]) * 3600)
hf169 = tensor153(S["x"][i169-1, 192:960]).reshape(-1, 4, 192)
hf169 = hf169 * torch.exp(-dt169[:, None, None] * torch.exp(W["core.flow.a"]))
z169 = tensor153(S["x"][i169, 960:992])
tokens_orig169 = torch.cat([
    hf169 + W["core.slots"], tensor153(e80[i169])[:, None],
    tensor153(u80[i169])[:, None], linear(z169, "z_in")[:, None]
], dim=1)
orig169 = prelude155(tokens_orig169)
mem169, mask169, age169 = memory82(i169, True)
age_emb169 = mlp(torch.log1p(age169 * 3600), "core.age")
hh169, oo169 = state_update82(hf169, orig169[0], mem169, mask169, age_emb169, 0)
print("Replay RMS h/out:",
      float(torch.mean((hh169-tensor153(S["x"][i169,192:960]).reshape(-1,4,192))**2).sqrt()),
      float(torch.mean((oo169-tensor153(S["x"][i169,:192]))**2).sqrt()))

print("Reserved output block norm:",
      float(zero169["core.prelude.0.attn.out.weight"][:,144:192].norm()))
print("Reserved KV value block norm:",
      float(zero169["core.prelude.0.attn.kv.weight"][336:384].norm()))
print("Candidate latent extractor factual mean/SD:")
ln169 = F.layer_norm(tensor153(raw_latent154), (192,),
                    W["core.prelude.0.norm1.weight"], W["core.prelude.0.norm1.bias"])
signal169 = ln169 @ tensor153(value_extract154)
print(signal169.mean(0), signal169.std(0))
print("Signal versus intended payload RMS:",
      torch.mean((signal169-tensor153(payload154))**2, dim=0).sqrt())

print("Scenario labels:")
for key, obj in list(globals().items()):
    if str(key).endswith("153") and isinstance(obj, (list, tuple)):
        print(key, repr(obj)[:500])
print("Scenario target column means:")
print(np.mean(y_cases153, axis=1))
with open("/work/notes.md", "a") as f:
    f.write("\n\nPython runs 165-169; forwards 10. Recovered architecture, record-derived effects, "
            "and state helpers. No additional edits; saved zero169 reversible transport checkpoint. "
            "Checking raw-seconds learned age keys, current-bundle memory, and latent payload units "
            "before a numerically calibrated transport pilot.\n")
```
